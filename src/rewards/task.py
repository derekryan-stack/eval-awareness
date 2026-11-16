"""Task correctness: 1.0 if the final answer (after </think>) matches the reference, else 0.0.

Unchanged behavior from the preliminary CoT-length experiment (metric: rewards/task_reward/mean).
"""

from grading import is_correct
from rewards.base import Reward, register


@register("task")
class TaskReward(Reward):
    metric_name = "task_reward"
    required_columns = ("answer",)

    def score(self, completions, prompts, answer, **columns):
        return [1.0 if c.finished_thinking and is_correct(c.response, ref) else 0.0 for c, ref in zip(completions, answer)]
