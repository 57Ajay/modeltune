"""Merge a LoRA adapter into its base model -> one standalone model (no PEFT needed to run it).

    python -m modeltune.merge CONFIG [--adapter X] [--push]

W_merged = W + (alpha / r) * B @ A. After merging, inference costs exactly what the base model
costs. Merge into the 16-bit base weights, never into a 4-bit model: re-quantizing after the
merge adds rounding error on top of the adapter (chapter 14). That's why we load on CPU in the
model's original dtype here, even if you trained with QLoRA.
"""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import cli, run_dir
from .evaluate import resolve_adapter
from .modeling import load_tokenizer


def main() -> None:
    def flags(p):
        p.add_argument("--adapter", default=None)
        p.add_argument("--push", action="store_true", help="push to <hub.repo_id>-merged")

    args, cfg = cli("Merge a LoRA adapter into the base model.", flags)
    adapter = resolve_adapter(cfg, False, args.adapter)
    m = cfg["model"]
    print(f"[modeltune] merging {adapter} into {m['id']} (on CPU, original dtype)")

    tok = load_tokenizer(cfg, adapter)
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype="auto",
                                                trust_remote_code=m["trust_remote_code"])
    base.generation_config.eos_token_id = tok.eos_token_id
    base.generation_config.pad_token_id = tok.pad_token_id
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()

    out = Path(run_dir(cfg)) / "merged"
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[modeltune] saved merged model to {out}")
    if args.push:
        repo = f"{cfg['hub']['repo_id']}-merged"
        merged.push_to_hub(repo, private=cfg["hub"]["private"])
        tok.push_to_hub(repo, private=cfg["hub"]["private"])
        print(f"[modeltune] pushed to https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
