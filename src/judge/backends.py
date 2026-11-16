"""Judge backends. Each turns a batch of chat-message lists into raw text outputs.

    hf         Local Hugging Face model on this machine's GPUs (default on Misha; no API needed).
               Any chat model checkpoint works, including OLMo-3.1-32B-Think itself or an Instruct model.
    anthropic  Claude via the official `anthropic` SDK (pip install anthropic). Credentials come from
               the environment (ANTHROPIC_API_KEY or an `ant auth login` profile); nothing is hard-coded.
    mock       Returns canned outputs; for tests and for dry-running the pipeline without a model.

Add a backend by subclassing Backend and registering it in BACKENDS.
"""

import json


class JudgeBackendError(RuntimeError):
    pass


class Backend:
    name = "base"

    def signature(self):
        """Everything that changes the judge's output; part of the cache key."""
        raise NotImplementedError

    def complete(self, batch):
        """batch: list of [{"role": ..., "content": ...}, ...]. Returns list of raw strings (same length)."""
        raise NotImplementedError


class MockBackend(Backend):
    """responses: a single string used for every call, or a list consumed in order (cycled)."""

    name = "mock"

    def __init__(self, responses='{"score": 0, "category": "no_awareness", "evidence": "", "explanation": "mock"}', **_):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = 0

    def signature(self):
        return {"backend": self.name, "responses": self.responses}

    def complete(self, batch):
        out = []
        for _ in batch:
            r = self.responses[self.calls % len(self.responses)]
            self.calls += 1
            if isinstance(r, Exception):
                raise r
            out.append(r)
        return out


class HFBackend(Backend):
    """Greedy decoding with a local checkpoint. Loaded lazily on first use."""

    name = "hf"

    def __init__(self, model_path, max_new_tokens=1024, batch_size=8, dtype="bfloat16", device_map="auto", **_):
        if not model_path:
            raise JudgeBackendError("hf judge backend needs model_path (set JUDGE_MODEL_PATH)")
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device_map = device_map
        self._model = self._tokenizer = None

    def signature(self):
        return {"backend": self.name, "model_path": self.model_path, "max_new_tokens": self.max_new_tokens, "decoding": "greedy"}

    def _load(self):
        if self._model is None:
            from modeling import load_model

            cfg = {"model": {"path": self.model_path, "dtype": self.dtype, "device_map": self.device_map,
                             "attn_implementation": None, "extra_eos_tokens": ["<|im_end|>"]}}
            self._model, self._tokenizer = load_model(cfg)
            self._model.eval()

    def complete(self, batch):
        from modeling import generate

        self._load()
        out = []
        for i in range(0, len(batch), self.batch_size):
            out += generate(self._model, self._tokenizer, batch[i:i + self.batch_size], max_new_tokens=self.max_new_tokens)
        return out


class AnthropicBackend(Backend):
    """Claude Messages API. Defaults to claude-opus-5-5; JSON is requested via the system prompt."""

    name = "anthropic"

    def __init__(self, model="claude-opus-5-5", max_tokens=4096, effort=None, fallbacks="default", client=None, **_):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.fallbacks = fallbacks
        self._client = client

    def signature(self):
        return {"backend": self.name, "model": self.model, "max_tokens": self.max_tokens, "effort": self.effort}

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise JudgeBackendError("anthropic judge backend needs the SDK: pip install anthropic") from e
            self._client = anthropic.Anthropic()  # resolves credentials from the environment
        return self._client

    def _one(self, messages):
        client = self._get_client()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        params = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[m for m in messages if m["role"] != "system"])
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            # Server-side fallback re-runs a safety-declined request on another model in the same call.
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks, **params)
        else:
            response = client.messages.create(**params)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise JudgeBackendError(f"judge model refused (category={getattr(details, 'category', None)})")
        return "".join(block.text for block in response.content if block.type == "text")

    def complete(self, batch):
        return [self._one(messages) for messages in batch]


BACKENDS = {"mock": MockBackend, "hf": HFBackend, "anthropic": AnthropicBackend}


def build_backend(spec):
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in BACKENDS:
        raise JudgeBackendError(f"Unknown judge backend {kind!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[kind](**spec)


def stable_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
