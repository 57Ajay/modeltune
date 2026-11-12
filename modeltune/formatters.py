"""Formatters: one raw row in, one standard-format row out (or None to skip the row).

Point `data.formatter` (and optionally `eval.formatter`) at one of these, e.g.
    data:
      formatter: modeltune.formatters:conll_to_ner_json

To use your own data, copy one of these, rename the columns, and you're done.
"""

from __future__ import annotations

import json
import re

# ----------------------------------------------------------------------------------------------
# Project 1: named-entity extraction as strict JSON (CoNLL-2003 -> prompt/completion)
# ----------------------------------------------------------------------------------------------
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
CONLL_KEYS = {"PER": "person", "ORG": "organization", "LOC": "location", "MISC": "misc"}

NER_INSTRUCTIONS = (
    "Extract all named entities from the user's text. Reply with ONLY a JSON object with exactly "
    'these keys: "person", "organization", "location", "misc". Each value is a list of the '
    "entity strings exactly as written in the text, in order of appearance. Use [] when there "
    "are none."
)


def detokenize(tokens: list[str]) -> str:
    """CoNLL text is pre-split into tokens ("U.S. , he said ."). Glue it back into normal text."""
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    return re.sub(r" (n't|'s|'re|'ve|'ll|'m|'d)\b", r"\1", text)


def conll_entities(tokens: list[str], tags: list[int]) -> dict[str, list[str]]:
    """Decode BIO tags into {"person": [...], "organization": [...], ...}."""
    entities: dict[str, list[str]] = {key: [] for key in CONLL_KEYS.values()}
    span: list[str] = []
    span_type = None

    def flush():
        if span:
            entities[CONLL_KEYS[span_type]].append(detokenize(span))

    for token, tag in zip(tokens, tags):
        label = CONLL_LABELS[tag]
        if label == "O":
            flush()
            span, span_type = [], None
        elif label.startswith("B-") or label[2:] != span_type:  # a new entity starts
            flush()
            span, span_type = [token], label[2:]
        else:  # I- continuing the current entity
            span.append(token)
    flush()
    return entities


def conll_to_ner_json(row: dict) -> dict | None:
    tokens = row["tokens"]
    if len(tokens) < 3 or tokens[0] == "-DOCSTART-":  # document separators / fragments
        return None
    target = conll_entities(tokens, row["ner_tags"])
    return {
        "prompt": [
            {"role": "system", "content": NER_INSTRUCTIONS},
            {"role": "user", "content": detokenize(tokens)},
        ],
        "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
    }


# ----------------------------------------------------------------------------------------------
# Project 2 evaluation: keep only the prompt of a preference pair
# ----------------------------------------------------------------------------------------------
def preference_to_prompt(row: dict) -> dict | None:
    """{"chosen": [user, assistant]} -> {"prompt": [user]} (for generating and scoring answers)."""
    if "prompt" in row and row["prompt"]:
        return {"prompt": row["prompt"]}
    return {"prompt": row["chosen"][:-1]}


# ----------------------------------------------------------------------------------------------
# Bonus: GSM8K math for GRPO (reasoning + verifiable answer)
# ----------------------------------------------------------------------------------------------
GSM8K_INSTRUCTIONS = (
    "Solve the math problem. First think step by step inside <reasoning>...</reasoning>, then "
    "write only the final number inside <answer>...</answer>."
)


def gsm8k(row: dict) -> dict:
    """GSM8K answers end with '#### 42'. Keep that number as the ground truth for the reward."""
    return {
        "prompt": [
            {"role": "system", "content": GSM8K_INSTRUCTIONS},
            {"role": "user", "content": row["question"]},
        ],
        "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
    }


# ----------------------------------------------------------------------------------------------
# Generic templates for your own data
# ----------------------------------------------------------------------------------------------
def alpaca(row: dict) -> dict:
    """{"instruction", "input", "output"} -> conversational prompt/completion."""
    user = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": row["output"]}],
    }


def sharegpt(row: dict) -> dict:
    """{"conversations": [{"from": "human"|"gpt"|"system", "value": ...}]} -> messages."""
    roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
    return {"messages": [{"role": roles[t["from"]], "content": t["value"]} for t in row["conversations"]]}
