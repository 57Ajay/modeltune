"""Generate answers on a held-out split and score them with the config's metric.

    python -m modeltune.evaluate CONFIG --base        # the model BEFORE fine-tuning (baseline)
    python -m modeltune.evaluate CONFIG               # after: uses outputs/<run_name> (or the Hub repo)
    python -m modeltune.evaluate CONFIG --adapter you/some-adapter

Always measure the baseline first. "It got better" means nothing without a "from what".
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from .config import cli, run_dir
from .data import load_eval_split
from .hardware import detect
from .inference import generate, load_for_inference
from .metrics import METRICS


def resolve_adapter(cfg: dict, base: bool, adapter: str | None) -> str | None:
    """Which weights to evaluate: --base (none), --adapter X, the local run, or the Hub repo."""
    if base:
        return None
    if adapter:
        return adapter
    local = run_dir(cfg)
    if Path(local, "adapter_config.json").exists():
        return local
    if cfg["hub"]["repo_id"]:
        return cfg["hub"]["repo_id"]
    raise SystemExit(f"No trained adapter found in {local}. Train first, pass --adapter, or use --base.")


def _reference(row: dict):
    if "completion" in row:
        c = row["completion"]
        return c[-1]["content"] if isinstance(c, list) else c
    return row.get("answer")


def main() -> None:
    def flags(p):
        p.add_argument("--base", action="store_true", help="evaluate the original model, no adapter")
        p.add_argument("--adapter", default=None, help="adapter path or Hub id to evaluate")

    args, cfg = cli("Evaluate a model on the config's held-out split.", flags)
    hw = detect()
    adapter = resolve_adapter(cfg, args.base, args.adapter)
    tag = "base" if args.base else "finetuned"
    print(f"[modeltune] evaluating {cfg['model']['id']} + adapter={adapter} on {hw}")

    e = cfg["eval"]
    ds = load_eval_split(cfg)
    prompts = ds["prompt"]
    references = [_reference(row) for row in ds]

    model, tok = load_for_inference(cfg, hw, adapter)
    predictions = generate(model, tok, prompts, e["max_new_tokens"], e["batch_size"], e["chat_template_kwargs"])
    del model  # free the GPU before a metric loads its own model (e.g. a reward model)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results = METRICS[e["metric"]](prompts, predictions, references, cfg, hw)
    results["avg_answer_words"] = sum(len(p.split()) for p in predictions) / max(len(predictions), 1)
    results["n"] = len(predictions)

    print(f"\n[modeltune] {tag} results:")
    for k, v in results.items():
        print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    print("\n[modeltune] examples:")
    for prompt, pred, ref in list(zip(prompts, predictions, references))[:3]:
        user = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        print(f"--- prompt: {user[:300]}\n>>> model : {pred[:500]}")
        if ref is not None:
            print(f">>> target: {str(ref)[:500]}")

    out = Path(run_dir(cfg))
    out.mkdir(parents=True, exist_ok=True)
    samples = [{"prompt": p, "prediction": y, "reference": r} for p, y, r in zip(prompts, predictions, references)]
    path = out / f"eval_{tag}.json"
    path.write_text(json.dumps({"adapter": adapter, "results": results, "samples": samples}, indent=2, ensure_ascii=False))
    print(f"\n[modeltune] saved {path}")


if __name__ == "__main__":
    main()
