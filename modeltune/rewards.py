"""Reward functions for GRPO (chapter 12).

TRL calls each function with the generated `completions` plus every extra dataset column as a
keyword argument (here: `answer`), and expects one float per completion. Rewards that a program
can check ("verifiable rewards") are what made reasoning RL practical: no reward model to hack.
"""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_RE = re.compile(r"\s*<reasoning>.+?</reasoning>\s*<answer>.+?</answer>\s*", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _text(completion) -> str:
    """Conversational completions are [{"role": "assistant", "content": ...}]."""
    return completion[0]["content"] if isinstance(completion, list) else completion


def _normalize(number: str) -> str:
    number = number.replace(",", "").strip()
    return number[:-2] if number.endswith(".0") else number


def extract_answer(text: str) -> str | None:
    """Prefer the <answer> tag; otherwise fall back to the last number in the text."""
    match = ANSWER_RE.search(text)
    numbers = NUMBER_RE.findall(match.group(1) if match else text)
    return _normalize(numbers[-1]) if numbers else None


def correctness(completions, answer, **kwargs) -> list[float]:
    """2.0 if the final number matches the ground truth, else 0.0. This is the real goal."""
    return [2.0 if extract_answer(_text(c)) == _normalize(a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    """0.5 for using the <reasoning>...</reasoning><answer>...</answer> structure. A small shaping
    bonus: it makes answers easy to parse, but it's worth less than being right."""
    return [0.5 if FORMAT_RE.fullmatch(_text(c)) else 0.0 for c in completions]
