"""Task metrics for `modeltune.evaluate` (chapter 08).

Every metric has the same signature: (prompts, predictions, references, cfg, hw) -> dict.
Loss tells you how well the model imitates the training data; these tell you whether it can
actually do the job.
"""

from __future__ import annotations

import json
import re

import torch

from . import rewards

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json(text: str):
    """Lenient parse: strip <think> blocks and ```json fences, then take the outermost {...}."""
    text = FENCE_RE.sub("", THINK_RE.sub("", text).strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def ner_json(prompts, predictions, references, cfg, hw) -> dict:
    strict = valid = exact = tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        gold = {(k, e) for k, v in json.loads(ref).items() for e in v}
        try:  # strict = the WHOLE output is valid JSON, nothing else (what a parser downstream needs)
            json.loads(pred)
            strict += 1
        except json.JSONDecodeError:
            pass
        obj = parse_json(pred)
        if not isinstance(obj, dict):
            fn += len(gold)
            continue
        valid += 1
        guess = {
            (k, e)
            for k, v in obj.items()
            if isinstance(v, list)
            for e in v
            if isinstance(e, str)
        }
        tp, fp, fn = tp + len(gold & guess), fp + len(guess - gold), fn + len(gold - guess)
        exact += gold == guess
    n = max(len(predictions), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "json_strict": strict / n,
        "json_parseable": valid / n,
        "exact_match": exact / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def gsm8k(prompts, predictions, references, cfg, hw) -> dict:
    n = max(len(predictions), 1)
    correct = sum(rewards.extract_answer(p) == rewards._normalize(r) for p, r in zip(predictions, references))
    formatted = sum(bool(rewards.FORMAT_RE.fullmatch(p)) for p in predictions)
    return {"accuracy": correct / n, "format_rate": formatted / n}


@torch.no_grad()
def reward_model(prompts, predictions, references, cfg, hw) -> dict:
    """Score each (prompt, answer) with a trained reward model, i.e. a model of human preference.
    A cheap stand-in for human or LLM-judge evaluation (and it can be gamed: watch the length)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rm_id = cfg["eval"]["reward_model"]
    tok = AutoTokenizer.from_pretrained(rm_id)
    device_map = {"": 0} if hw.device == "cuda" else None
    rm = AutoModelForSequenceClassification.from_pretrained(rm_id, dtype=hw.dtype, device_map=device_map)
    scores = []
    for prompt, pred in zip(prompts, predictions):
        conv = list(prompt) + [{"role": "assistant", "content": pred}]
        text = tok.apply_chat_template(conv, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
        scores.append(rm(**ids.to(rm.device)).logits[0][0].float().item())
    return {"reward_mean": sum(scores) / max(len(scores), 1)}


def none(prompts, predictions, references, cfg, hw) -> dict:
    return {}


METRICS = {"none": none, "ner_json": ner_json, "gsm8k": gsm8k, "reward_model": reward_model}
