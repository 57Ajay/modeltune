"""Load any dataset and convert it into one of the standard formats TRL understands.

    SFT   {"messages": [...]}                       conversational (multi-turn chat)
          {"prompt": [...], "completion": [...]}    conversational prompt/completion
          {"text": "..."}                           plain text (continued pre-training)
    DPO   {"prompt": [...], "chosen": [...], "rejected": [...]}   (prompt may be implicit)
    GRPO  {"prompt": [...], ...extra columns your reward functions read...}

If your raw data has other columns, write a small formatter, a function that takes one row
and returns one of the dicts above (or None to drop the row), and point `data.formatter` at
it. Examples are in `modeltune/formatters.py`. That's the whole "bring your own data" story.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from datasets import Dataset, load_dataset

LOCAL_FORMATS = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}


def import_function(path: str):
    """'package.module:function' -> the function object."""
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _load_raw(dataset: str, subset: str | None, revision: str | None, split: str) -> Dataset:
    suffix = Path(dataset).suffix
    if suffix in LOCAL_FORMATS:  # local file: everything is in the "train" split
        return load_dataset(LOCAL_FORMATS[suffix], data_files=dataset, split="train")
    return load_dataset(dataset, subset, revision=revision, split=split)


def _add_system_prompt(row: dict, system_prompt: str) -> dict:
    for key in ("messages", "prompt"):
        conv = row.get(key)
        if isinstance(conv, list) and conv and conv[0]["role"] != "system":
            row[key] = [{"role": "system", "content": system_prompt}] + conv
    return row


def prepare(
    ds: Dataset,
    formatter: str | None = None,
    max_samples: int | None = None,
    system_prompt: str | None = None,
    seed: int = 42,
) -> Dataset:
    if formatter:
        fn = import_function(formatter)
        ds = ds.filter(lambda row: fn(row) is not None)  # formatter returns None -> drop row
        ds = ds.map(fn, remove_columns=ds.column_names)
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))  # random, but reproducible
    if system_prompt:
        ds = ds.map(lambda row: _add_system_prompt(row, system_prompt))
    return ds


def load_train_eval(cfg: dict) -> tuple[Dataset, Dataset | None]:
    d, seed = cfg["data"], cfg["seed"]
    train = _load_raw(d["dataset"], d["subset"], d["revision"], d["train_split"])
    eval_ds = None
    if isinstance(d["eval_split"], float):  # carve a held-out slice out of train
        parts = train.train_test_split(test_size=d["eval_split"], seed=seed)
        train, eval_ds = parts["train"], parts["test"]
    elif d["eval_split"]:
        eval_ds = _load_raw(d["dataset"], d["subset"], d["revision"], d["eval_split"])

    train = prepare(train, d["formatter"], d["max_train_samples"], d["system_prompt"], seed)
    if eval_ds is not None:
        eval_ds = prepare(eval_ds, d["formatter"], d["max_eval_samples"], d["system_prompt"], seed)
    return train, eval_ds


def load_eval_split(cfg: dict) -> Dataset:
    """The held-out split used by `modeltune.evaluate` (defaults to the training dataset's source)."""
    e, d = cfg["eval"], cfg["data"]
    if e["dataset"]:  # a different evaluation dataset: use its own subset/revision/formatter
        source = (e["dataset"], e["subset"], e["revision"])
        formatter = e["formatter"]
    else:
        source = (d["dataset"], d["subset"], d["revision"])
        formatter = e["formatter"] or d["formatter"]
    ds = _load_raw(*source, e["split"])
    return prepare(ds, formatter, e["max_samples"], d["system_prompt"], cfg["seed"])
