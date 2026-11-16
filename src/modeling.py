"""Model, tokenizer, and LoRA setup shared by training, the smoke test, and the judge's local backend."""

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ModelSetupError(RuntimeError):
    pass


def load_tokenizer(model_path, padding_side="left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side=padding_side)
    if tok.pad_token is None:
        # OLMo 3 defines <|pad|>; this is only a fallback for other checkpoints.
        tok.pad_token = tok.eos_token
    if not tok.chat_template:
        raise ModelSetupError(f"Tokenizer at {model_path} has no chat template; a thinking model needs one")
    return tok


def check_thinking_template(tokenizer):
    """Return the generation prompt suffix and whether it opens a <think> block (OLMo 3 Think does)."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    return {"generation_prompt_ends_with_think": text.rstrip().endswith("<think>"), "generation_prompt_tail": text[-40:]}


def stop_token_ids(tokenizer, generation_config, extra_tokens=()):
    """eos ids from generation_config plus any extra stop tokens that exist in the vocabulary."""
    eos = generation_config.eos_token_id if generation_config is not None else None
    ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    unk = tokenizer.unk_token_id
    for tok in extra_tokens or ():
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != unk and tid not in ids:
            ids.append(tid)
    return ids


def load_model(cfg, tokenizer=None):
    """Load the base model (bf16, sharded with device_map) and its tokenizer."""
    from transformers import AutoModelForCausalLM

    m = cfg["model"]
    path = m["path"]
    tokenizer = tokenizer or load_tokenizer(path)
    kwargs = dict(dtype=DTYPES[m["dtype"]], device_map=m["device_map"], local_files_only=True)
    if m.get("attn_implementation"):
        kwargs["attn_implementation"] = m["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    # OLMo 3's config.json sets use_cache=False; generation explicitly re-enables the KV cache.
    model.generation_config.eos_token_id = stop_token_ids(tokenizer, model.generation_config, m.get("extra_eos_tokens"))
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def validate_lora_targets(model, target_modules):
    """Every configured target must match at least one module name suffix (PEFT only errors if none match)."""
    if target_modules == "all-linear":
        return {"all-linear": None}
    names = [n for n, _ in model.named_modules()]
    counts = {t: sum(1 for n in names if n == t or n.endswith("." + t)) for t in target_modules}
    missing = [t for t, c in counts.items() if c == 0]
    if missing:
        leaf_types = sorted({n.rsplit(".", 1)[-1] for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)})
        raise ModelSetupError(f"LoRA target_modules {missing} match no module in the model. Linear layers present: {leaf_types}")
    return counts


def build_lora_config(cfg):
    from peft import LoraConfig

    l = cfg["lora"]
    return LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"], target_modules=l["target_modules"], task_type="CAUSAL_LM")


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens, temperature=0.0, top_p=1.0, system_prompt=None):
    """Batch-generate one completion per prompt (str or chat messages). Returns decoded completions."""
    texts = []
    for p in prompts:
        messages = p if isinstance(p, list) else ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": p}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    finally:
        tokenizer.padding_side = old_side
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True, **sampling)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in out[:, inputs["input_ids"].shape[1]:]]


def describe_checkpoint(path):
    """Identifiers for metadata: config architecture and a fingerprint of the weight index."""
    import hashlib
    import json

    p = Path(path)
    info = {"path": str(p.resolve()) if p.exists() else str(p)}
    cfg_file = p / "config.json"
    if cfg_file.exists():
        c = json.loads(cfg_file.read_text())
        info.update(architectures=c.get("architectures"), model_type=c.get("model_type"), name_or_path=c.get("_name_or_path"))
    for name in ("model.safetensors.index.json", "model.safetensors", "config.json"):
        f = p / name
        if f.exists() and f.stat().st_size < 50_000_000:
            info["fingerprint"] = {name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]}
            break
    return info
