"""Evaluation-awareness rewards that use the LLM judge (src/judge/).

Two registered rewards share one implementation and differ only in sign:

  awareness_penalty  reward = -judge_score_normalized   (discourage verbalized eval awareness)
  awareness_reward   reward = +judge_score_normalized   (encourage verbalized eval awareness)

The judge score is 0-3 normalized to 0-1. A failed judge call gives None (ignored by TRL, share
logged as rewards/<metric>/invalid_frac). Config:

  - name: awareness_penalty
    weight: 1.0
    judge_config: configs/judge.yaml   # backend etc.; JUDGE_* env vars override
    target: full                        # full | reasoning | response (what the judge reads)

NOTE: with the default `hf` backend the judge is a second model loaded inside the training
process; give the job enough GPU memory (or use the anthropic backend, or a smaller JUDGE_MODEL_PATH).
"""

from judge.judge import build_judge, load_judge_config
from rewards.base import Reward, register


class _AwarenessReward(Reward):
    sign = 1.0

    def __init__(self, ctx, label=None, judge_config="configs/judge.yaml", target=None, judge=None):
        super().__init__(ctx, label=label)
        if judge is None:
            from config import REPO_ROOT

            overrides = {"target": target}
            if ctx.run_dir is not None:
                overrides["log_path"] = str(ctx.run_dir / f"judge_{self.__name__}.log.jsonl")
            path = REPO_ROOT / judge_config
            judge = build_judge(load_judge_config(path if path.exists() else judge_config, overrides=overrides))
        self.judge = judge

    def score(self, completions, prompts, **columns):
        items = [{"prompt": p, "reasoning": c.reasoning, "response": c.response, "finished_thinking": c.finished_thinking}
                 for p, c in zip(prompts, completions)]
        results = self.judge.judge_batch(items)
        valid = [r.score for r in results if r.valid]
        if valid:
            self.log_metric("awareness/judge_score_mean", sum(valid) / len(valid))
            self.log_metric("awareness/frac_score_ge2", sum(s >= 2 for s in valid) / len(valid))
        return [None if not r.valid else self.sign * r.normalized for r in results]


@register("awareness_penalty")
class AwarenessPenalty(_AwarenessReward):
    metric_name = "awareness_penalty"
    sign = -1.0


@register("awareness_reward")
class AwarenessReward(_AwarenessReward):
    metric_name = "awareness_reward"
    sign = 1.0
