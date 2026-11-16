"""Chain-of-thought length penalty, as used in the preliminary experiment.

reward = -coef * min(reasoning_tokens, max_tokens) / max_tokens, where reasoning is everything
before </think>; unfinished reasoning counts as max_tokens. Logs cot_tokens/mean (read by analyze.py).
"""

from rewards.base import Reward, RewardConfigError, register


@register("cot_length")
class CotLengthPenalty(Reward):
    metric_name = "cot_length_penalty"

    def __init__(self, ctx, label=None, coef=1.0, max_tokens=None):
        super().__init__(ctx, label=label)
        if ctx.tokenizer is None:
            raise RewardConfigError("cot_length needs a tokenizer to count reasoning tokens")
        self.coef = float(coef)
        self.max_tokens = int(max_tokens or ctx.max_completion_length)

    def score(self, completions, prompts, **columns):
        rewards, lengths = [], []
        for c in completions:
            n = len(self.ctx.tokenizer(c.reasoning, add_special_tokens=False)["input_ids"]) if c.finished_thinking else self.max_tokens
            lengths.append(n)
            rewards.append(-self.coef * min(n, self.max_tokens) / self.max_tokens)
        self.log_metric("cot_tokens/mean", sum(lengths) / len(lengths))
        return rewards
