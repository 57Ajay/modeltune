"""Load the tokenizer and model exactly the way the config says.

Three decisions happen here (chapters 04-06):
1. Precision: 16-bit weights on GPU, or 4-bit NF4 weights for QLoRA.
2. Chat template: instruct models ship one; base models usually don't, so we provide one.
3. Adapters: start from scratch (TRL adds a fresh LoRA) or continue from an existing adapter.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hardware import Hardware

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_template(value: str) -> str:
    """`value` is a .jinja file (relative to cwd or repo root) or a tokenizer id to copy from."""
    if value.endswith(".jinja"):
        path = Path(value) if Path(value).exists() else REPO_ROOT / value
        return path.read_text()
    return AutoTokenizer.from_pretrained(value).chat_template


def load_tokenizer(cfg: dict, adapter: str | None = None):
    m = cfg["model"]
    tok = None
    if adapter:  # an adapter saved by our trainer also contains the tokenizer it was trained with
        try:
            tok = AutoTokenizer.from_pretrained(adapter)
        except (OSError, ValueError):
            tok = None
    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            m["id"], revision=m["revision"], trust_remote_code=m["trust_remote_code"]
        )
    if m["chat_template"]:
        tok.chat_template = _read_template(m["chat_template"])
    if m["eos_token"]:
        # The token the model emits to end its turn. If this is wrong the model never stops talking.
        if m["eos_token"] not in tok.get_vocab():
            raise ValueError(
                f"eos_token {m['eos_token']!r} is not in {m['id']}'s vocabulary. Use a template built "
                "from tokens the model already has (e.g. Llama 3 uses <|eot_id|>); new tokens would "
                "need new, trained embeddings (see course/06)."
            )
        tok.eos_token = m["eos_token"]
    if tok.pad_token is None:
        # Padding positions are excluded from the loss via the attention mask, so reusing EOS is safe.
        tok.pad_token = tok.eos_token
    return tok


def _quantization(kind: str, hw: Hardware):
    if kind in (None, "none"):
        return None
    if hw.device != "cuda":
        raise ValueError("bitsandbytes 4-bit/8-bit quantization needs a CUDA GPU")
    if kind == "4bit":
        # QLoRA recipe: NF4 storage + double quantization + 16-bit compute (chapter 05).
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=hw.dtype,
        )
    if kind == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unknown quantization: {kind}")


def load_model(cfg: dict, hw: Hardware, tokenizer, adapter: str | None = None, trainable: bool = True):
    m = cfg["model"]
    full_finetune = cfg["lora"] is None and not adapter
    # fp16 mixed precision keeps fp32 "master" weights for whatever is trained. With LoRA, PEFT
    # already keeps the small adapter in fp32; with full fine-tuning the whole model must be fp32.
    dtype = torch.float32 if (full_finetune and hw.fp16) else hw.dtype
    device_map = {"": int(os.environ.get("LOCAL_RANK", 0))} if hw.device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        m["id"],
        revision=m["revision"],
        dtype=dtype,
        quantization_config=_quantization(m["quantization"], hw),
        attn_implementation=m["attn_implementation"],
        trust_remote_code=m["trust_remote_code"],
        device_map=device_map,
    )
    if hw.device == "mps":
        model.to("mps")

    # MoE naming differs across families (Mixtral/Granite: num_local_experts, Qwen-MoE/OLMoE:
    # num_experts). TRL's load-balancing loss reads `num_experts`, so make sure it exists.
    text_config = model.config.get_text_config()
    if hasattr(text_config, "num_local_experts") and not hasattr(text_config, "num_experts"):
        text_config.num_experts = text_config.num_local_experts

    # Keep model and tokenizer in agreement about special tokens (important for base models).
    for conf in (text_config, model.generation_config):
        conf.eos_token_id = tokenizer.eos_token_id
        conf.pad_token_id = tokenizer.pad_token_id

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model


def lora_config(cfg: dict) -> LoraConfig | None:
    """A fresh LoRA config, or None for full fine-tuning / when continuing an existing adapter."""
    lora = cfg["lora"]
    if lora is None or cfg["model"]["adapter"]:
        return None
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        target_parameters=lora["target_parameters"],
        task_type="CAUSAL_LM",
    )
