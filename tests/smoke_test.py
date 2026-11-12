"""End-to-end smoke test on CPU with tiny random models (~5 min, no GPU needed).

    uv run python tests/smoke_test.py

It runs every real config in configs/ with a handful of steps, swapping in tiny test models via
the same key=value overrides you'd use yourself. It proves the plumbing works (data -> template ->
trainer -> save -> eval -> merge); it says nothing about quality. Run it after you change the code.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_MOE = "trl-internal-testing/tiny-GraniteMoeForCausalLM"


def run(module: str, config: str, *args: str) -> None:
    cmd = [sys.executable, "-m", f"modeltune.{module}", f"configs/{config}", *args]
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    out = tempfile.mkdtemp(prefix="modeltune-smoke-")
    common = [f"output_dir={out}", "hub.push=false", "train.max_steps=2", "train.logging_steps=1",
              "train.eval_steps=1", "data.max_train_samples=16", "data.max_eval_samples=4",
              "eval.max_samples=4", "eval.max_new_tokens=16", "eval.batch_size=2"]
    cpu = ["model.quantization=none", "train.optim=adamw_torch"]

    # Project 1: prompt/completion SFT + LoRA, eval before/after, chat, merge
    p1 = "p1_qwen3_ner_json_lora.yaml"
    run("train", p1, f"model.id={TINY}", *common, "train.per_device_train_batch_size=4")
    run("evaluate", p1, f"model.id={TINY}", *common, "--base")
    run("evaluate", p1, f"model.id={TINY}", *common)
    run("chat", p1, f"model.id={TINY}", *common, "--prompt", "Angela Merkel met Google in Paris.", "--max_new_tokens", "8")
    run("merge", p1, f"model.id={TINY}", *common)

    # Project 2: base-model chat template + assistant-only SFT, then DPO from that adapter
    sft, dpo = "p2_smollm3_sft_qlora.yaml", "p2_smollm3_dpo_qlora.yaml"
    run("train", sft, f"model.id={TINY}", *common, *cpu)
    sft_adapter = f"{out}/p2-smollm3-3b-sft-qlora"
    run("train", dpo, f"model.id={TINY}", f"model.adapter={sft_adapter}", *common, *cpu)
    run("evaluate", dpo, f"model.id={TINY}", f"model.adapter={sft_adapter}", *common, *cpu, "eval.metric=none")

    # Bonus A: GRPO with programmatic rewards
    grpo = "bonus_grpo_qwen3_gsm8k.yaml"
    run("train", grpo, f"model.id={TINY}", *common, "train.max_steps=1", "train.num_generations=2",
        "train.per_device_train_batch_size=2", "train.gradient_accumulation_steps=1", "train.max_completion_length=8")
    run("evaluate", grpo, f"model.id={TINY}", *common)

    # Bonus B: MoE with LoRA on fused expert tensors + router load-balancing loss
    moe = "bonus_moe_granite_ner_json_lora.yaml"
    run("train", moe, f"model.id={TINY_MOE}", *common, "train.per_device_train_batch_size=2")
    run("evaluate", moe, f"model.id={TINY_MOE}", *common)

    print(f"\nSMOKE TEST PASSED (artifacts in {out})")


if __name__ == "__main__":
    main()
