"""One YAML file = one training run.

Every command in this package (train / evaluate / chat / merge) takes the same YAML file, so
"fine-tune a different model" or "fine-tune on different data" means editing one line, or
overriding it from the command line:

    python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml model.id=Qwen/Qwen3-1.7B

Your YAML only needs the keys you want to change; everything else comes from DEFAULTS below.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run_name": None,  # defaults to the YAML file name
    "method": "sft",  # sft | dpo | grpo
    "output_dir": "outputs",  # checkpoints go to <output_dir>/<run_name>
    "seed": 42,
    "model": {
        "id": None,  # any Hugging Face causal LM id, e.g. Qwen/Qwen3-0.6B
        "revision": None,
        "quantization": "none",  # none | 4bit (QLoRA) | 8bit
        "adapter": None,  # start from an existing LoRA adapter (local path or Hub id)
        "chat_template": None,  # .jinja file or a tokenizer id to copy the template from
        "eos_token": None,  # end-of-turn token; base models usually need this set
        "attn_implementation": "sdpa",  # sdpa works on every GPU; flash_attention_2 needs Ampere+
        "trust_remote_code": False,
    },
    # LoRA settings. Set `lora: null` in your YAML for full fine-tuning.
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "all-linear",
        "target_parameters": None,  # for MoE experts stored as raw tensors (see chapter 13)
    },
    "data": {
        "dataset": None,  # Hub id or local .jsonl/.json/.csv/.parquet file
        "subset": None,  # a.k.a. dataset "config name"
        "revision": None,
        "train_split": "train",
        "eval_split": None,  # a split name, or a float like 0.05 to carve it out of train
        "formatter": None,  # "module:function" that maps one raw row -> standard format
        "max_train_samples": None,
        "max_eval_samples": None,
        "system_prompt": None,  # prepended to every conversation that has none
    },
    # Passed straight into TRL's SFTConfig / DPOConfig / GRPOConfig. Anything those accept works.
    "train": {},
    # GRPO only: list of "module:function" reward functions.
    "rewards": [],
    # Task evaluation (python -m modeltune.evaluate). Dataset keys default to the `data` section.
    "eval": {
        "metric": "none",  # none | ner_json | gsm8k | reward_model
        "dataset": None,
        "subset": None,
        "revision": None,
        "split": "test",
        "formatter": None,
        "max_samples": 200,
        "max_new_tokens": 256,
        "batch_size": 16,
        "chat_template_kwargs": {},  # e.g. {enable_thinking: false} for Qwen3
        "reward_model": None,  # used by metric: reward_model
    },
    "hub": {
        "push": False,
        "repo_id": None,  # e.g. your-name/qwen3-0.6b-ner-json-lora
        "private": False,
    },
}


def _merge(base: dict, new: dict) -> dict:
    """Recursively overlay `new` on top of `base`."""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str, overrides: list[str] | tuple[str, ...] = ()) -> dict:
    """Read YAML, fill defaults, apply `a.b.c=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(copy.deepcopy(DEFAULTS), user)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            if node.get(p) is None:
                node[p] = {}
            node = node[p]
        node[leaf] = yaml.safe_load(raw)
    cfg["run_name"] = cfg["run_name"] or Path(path).stem
    if not cfg["model"]["id"]:
        raise ValueError("model.id is required")
    return cfg


def run_dir(cfg: dict) -> str:
    return str(Path(cfg["output_dir"]) / cfg["run_name"])


def cli(description: str, extra_args=None) -> tuple[argparse.Namespace, dict]:
    """Shared command line: CONFIG [--flags] [key=value ...]."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="path to a YAML run config")
    if extra_args:
        extra_args(parser)
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if o.startswith("-") or "=" not in o]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)} (overrides look like key.sub=value)")
    return args, load_config(args.config, overrides)
