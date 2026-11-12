"""Fine-tune any causal LM with SFT, DPO or GRPO, driven by one YAML config.

    python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml
    python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml model.id=Qwen/Qwen3-1.7B train.learning_rate=1e-4

The flow is the same for every method: hardware -> tokenizer -> model -> data -> trainer -> save.
Only the trainer class and its config change.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from transformers import set_seed
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from .config import cli, run_dir
from .data import import_function, load_train_eval
from .hardware import detect, peak_memory_gb
from .modeling import load_model, load_tokenizer, lora_config

METHODS = {
    "sft": (SFTConfig, SFTTrainer),  # imitate demonstrations
    "dpo": (DPOConfig, DPOTrainer),  # prefer chosen over rejected answers
    "grpo": (GRPOConfig, GRPOTrainer),  # reinforce answers that score well on a reward
}


def build_args(cfg: dict, hw):
    config_cls = METHODS[cfg["method"]][0]
    kwargs = {
        "output_dir": run_dir(cfg),
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "bf16": hw.bf16,  # Ampere+ GPUs
        "fp16": hw.fp16,  # T4 / V100: fp16 autocast + loss scaling
        "report_to": "none",
        "logging_steps": 10,
        "save_strategy": "no",  # we save once at the end; set "steps" in YAML for checkpoints
    }
    kwargs.update(cfg["train"])  # your YAML always wins
    if cfg["hub"]["push"]:
        kwargs.update(
            push_to_hub=True,
            hub_model_id=cfg["hub"]["repo_id"],
            hub_private_repo=cfg["hub"]["private"],
            hub_strategy="end",
        )
    return config_cls(**kwargs)


def show_example(trainer, tok) -> None:
    """Print one training example exactly as the model sees it, and which part it learns from.
    The best debugging habit in fine-tuning: always look at your tokenized data."""
    row = trainer.train_dataset[0]
    if "input_ids" not in row:
        return
    ids = row["input_ids"]
    if "labels" in row:  # -100 = "ignore this position in the loss"
        mask = [label != -100 for label in row["labels"]]
    else:
        mask = row.get("completion_mask") or row.get("assistant_masks") or [1] * len(ids)
    learned = [i for i, m in zip(ids, mask) if m]
    print("\n" + "=" * 30 + " one tokenized training example " + "=" * 30)
    print(tok.decode(ids))
    print("-" * 30 + f" loss is computed on {len(learned)}/{len(ids)} tokens: " + "-" * 30)
    print(tok.decode(learned))
    print("=" * 92 + "\n")


def main() -> None:
    _, cfg = cli("Fine-tune a model from a YAML config.")
    set_seed(cfg["seed"])
    hw = detect()
    print(f"[modeltune] hardware: {hw}")
    print(f"[modeltune] run: {cfg['run_name']}  method={cfg['method']}  model={cfg['model']['id']}")

    tok = load_tokenizer(cfg, cfg["model"]["adapter"])
    model = load_model(cfg, hw, tok, adapter=cfg["model"]["adapter"], trainable=True)
    train_ds, eval_ds = load_train_eval(cfg)
    print(f"[modeltune] data: {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval rows")

    config_cls, trainer_cls = METHODS[cfg["method"]]
    kwargs = dict(
        model=model,
        args=build_args(cfg, hw),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    if cfg["method"] == "grpo":
        kwargs["reward_funcs"] = [import_function(f) for f in cfg["rewards"]]
    trainer = trainer_cls(**kwargs)

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    if cfg["method"] == "sft":
        show_example(trainer, tok)

    start = time.time()
    # Colab disconnected? Set train.save_strategy=steps (+ save_steps) and re-run with
    # train.resume_from_checkpoint=true (latest checkpoint in output_dir) or =path/to/checkpoint-N.
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
    minutes = (time.time() - start) / 60
    print(f"[modeltune] trained in {minutes:.1f} min, peak GPU memory {peak_memory_gb():.2f} GB")

    out = run_dir(cfg)
    trainer.save_model(out)  # LoRA: only the adapter (MBs). Full fine-tune: the whole model.
    Path(out, "modeltune_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if eval_ds is not None:
        print(f"[modeltune] final eval: {trainer.evaluate()}")
    if cfg["hub"]["push"]:
        trainer.push_to_hub(dataset_name=cfg["data"]["dataset"])
        print(f"[modeltune] pushed to https://huggingface.co/{cfg['hub']['repo_id']}")
    print(f"[modeltune] saved to {out}")


if __name__ == "__main__":
    main()
