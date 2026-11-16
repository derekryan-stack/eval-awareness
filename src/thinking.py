"""Splitting thinking-model completions into reasoning and final answer.

OLMo 3.1 Think's chat template ends the generation prompt with "<|im_start|>assistant\\n<think>",
so a completion is "<reasoning></think><response>". "</think>" is ordinary text (three BPE
tokens, not a special token), so it survives decoding with skip_special_tokens=True. If the
model itself emits a leading "<think>" (e.g. with a different template), it is stripped.

No heavy imports here: rewards, the judge, and analysis code all share this module.
"""

from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ParsedCompletion:
    reasoning: str
    response: str
    finished_thinking: bool  # False: generation stopped (usually max length) before </think>


def completion_text(completion):
    """TRL passes chat-format completions as [{"role": "assistant", "content": ...}]; plain prompts give str."""
    if isinstance(completion, list):
        return "".join(m.get("content") or "" for m in completion if m.get("role", "assistant") == "assistant")
    if isinstance(completion, dict):
        return completion.get("content") or ""
    return completion or ""


def split_completion(text):
    """Return ParsedCompletion. Everything before the first </think> is reasoning.

    Without </think> the whole text is (unfinished) reasoning and the response is empty, so a
    truncated chain of thought is never graded as a final answer.
    """
    text = completion_text(text)
    stripped = text.lstrip()
    if stripped.startswith(THINK_OPEN):
        text = stripped[len(THINK_OPEN):]
    reasoning, sep, response = text.partition(THINK_CLOSE)
    if not sep:
        return ParsedCompletion(reasoning=text, response="", finished_thinking=False)
    return ParsedCompletion(reasoning=reasoning, response=response, finished_thinking=True)
