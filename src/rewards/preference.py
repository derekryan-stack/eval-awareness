"""Generic preference reward from an off-the-shelf reward model (sequence-classification head).

Scores the user prompt + final answer (reasoning is not shown to the reward model). Completions
that never finished thinking have no answer and get None. Raw RM logits are returned; GRPO
normalizes within each group, and `scale` can shrink them when mixing with other rewards.

  - name: preference
    weight: 1.0
    model_path: /path/to/reward-model   # or PREFERENCE_RM_PATH; HF-format AutoModelForSequenceClassification
    scale: 1.0
    batch_size: 4
"""

import os

from rewards.base import Reward, RewardConfigError, register


@register("preference")
class PreferenceReward(Reward):
    metric_name = "preference_reward"

    def __init__(self, ctx, label=None, model_path=None, scale=1.0, batch_size=4, max_length=4096, dtype="bfloat16"):
        super().__init__(ctx, label=label)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from config import REPO_ROOT

        path = model_path or os.environ.get("PREFERENCE_RM_PATH")
        if not path:
            raise RewardConfigError("preference reward needs model_path or the PREFERENCE_RM_PATH environment variable")
        if not os.path.isabs(path) and (REPO_ROOT / path).exists():
            path = str(REPO_ROOT / path)
        self.scale, self.batch_size, self.max_length = float(scale), int(batch_size), int(max_length)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.model.eval()

    def _text(self, prompt, response):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        chat = [m for m in prompt if m.get("role") != "system"] + [{"role": "assistant", "content": response}]
        return self.tok.apply_chat_template(chat, tokenize=False)

    def score(self, completions, prompts, **columns):
        import torch

        idx = [i for i, c in enumerate(completions) if c.finished_thinking and c.response.strip()]
        out = [None] * len(completions)
        for s in range(0, len(idx), self.batch_size):
            batch = idx[s : s + self.batch_size]
            texts = [self._text(prompts[i], completions[i].response) for i in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, 0].float().cpu().tolist()
            for i, v in zip(batch, logits):
                out[i] = self.scale * v
        return out
