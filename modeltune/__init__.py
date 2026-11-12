"""modeltune: a small, readable fine-tuning toolkit that goes with the course in `course/`.

Modules, in the order data flows through them:
    config      YAML run config + defaults + key=value overrides
    hardware    device and precision detection (T4 fp16 vs Ampere+ bf16)
    data        load any dataset -> standard format (formatters.py has examples)
    modeling    tokenizer, model, quantization, chat template, LoRA config
    train       SFT / DPO / GRPO with TRL
    inference   batched generation
    evaluate    task metrics (metrics.py) before vs after
    chat        talk to the model
    merge       fold a LoRA adapter into the base weights
"""

__version__ = "1.0.0"
