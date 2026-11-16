"""Shared fixtures. Tests run offline on CPU: the "model" is a tiny randomly initialized OLMo 3
(the same architecture class as OLMo-3.1-32B-Think) with a small BPE tokenizer that uses the real
OLMo 3 Think chat template (tests/fixtures/olmo3_think_chat_template.jinja)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"

SPECIAL = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


def build_tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "What is 7 * 6? Answer with just the number. <think> Let me think. </think> The answer is 42.",
        "You are Olmo, a helpful AI assistant built by Ai2. system user assistant \\boxed{11/16}",
        "A fair coin is flipped 4 times. What is the probability of getting at least 2 heads?",
    ] * 20
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                   unk_token="<|endoftext|>", pad_token="<|pad|>")
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    fast.chat_template = (FIXTURES / "olmo3_think_chat_template.jinja").read_text()
    return fast


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return build_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_tokenizer):
    import torch
    from transformers import GenerationConfig, Olmo3Config, Olmo3ForCausalLM

    d = tmp_path_factory.mktemp("tiny-olmo3-think")
    torch.manual_seed(0)
    cfg = Olmo3Config(vocab_size=len(tiny_tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512, sliding_window=64,
                      layer_types=["sliding_attention", "full_attention"], pad_token_id=tiny_tokenizer.pad_token_id,
                      eos_token_id=tiny_tokenizer.eos_token_id, bos_token_id=tiny_tokenizer.eos_token_id, use_cache=False)
    model = Olmo3ForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=tiny_tokenizer.eos_token_id, pad_token_id=tiny_tokenizer.pad_token_id,
                                               do_sample=True, temperature=0.6, top_p=0.95)
    model.save_pretrained(d)
    tiny_tokenizer.save_pretrained(d)
    return d


@pytest.fixture
def trial_jsonl(tmp_path):
    rows = [{"prompt": f"What is {i} + {i}? Answer with just the number.", "answer": str(2 * i), "family": "add"} for i in range(6)]
    p = tmp_path / "trial.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture
def clean_env(monkeypatch):
    for var in ["MODEL_PATH", "DATASET_PATH", "EVAL_DATASET_PATH", "OUTPUT_ROOT", "OUTPUT_DIR", "EXPERIMENT_NAME", "SEED", "SLURM_JOB_ID", "WORLD_SIZE"]:
        monkeypatch.delenv(var, raising=False)
