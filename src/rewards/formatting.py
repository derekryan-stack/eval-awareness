"""Orthogonal control: a formatting reward unrelated to correctness and to evaluation awareness.

reward = 1.0 if thinking finished and the final answer contains exactly one \\boxed{...} and is
short (<= max_response_chars); partial credit 0.5 for a single \\boxed{...} that is too long;
otherwise 0.0.

  - name: formatting
    weight: 1.0
    max_response_chars: 400
"""

import re

from rewards.base import Reward, register

BOXED = re.compile(r"\\boxed\{")


@register("formatting")
class FormattingReward(Reward):
    metric_name = "formatting_reward"

    def __init__(self, ctx, label=None, max_response_chars=400):
        super().__init__(ctx, label=label)
        self.max_chars = int(max_response_chars)

    def score(self, completions, prompts, **columns):
        out = []
        for c in completions:
            if not c.finished_thinking or len(BOXED.findall(c.response)) != 1:
                out.append(0.0)
            else:
                out.append(1.0 if len(c.response) <= self.max_chars else 0.5)
        return out
