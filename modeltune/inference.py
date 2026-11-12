"""Batched generation shared by `evaluate` and `chat`."""

from __future__ import annotations

import torch

from .hardware import Hardware
from .modeling import load_model, load_tokenizer


def load_for_inference(cfg: dict, hw: Hardware, adapter: str | None):
    tok = load_tokenizer(cfg, adapter)
    # Decoder-only models generate to the right, so the batch must be padded on the LEFT;
    # otherwise shorter prompts would have pad tokens between the prompt and the answer.
    tok.padding_side = "left"
    model = load_model(cfg, hw, tok, adapter=adapter, trainable=False)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts: list, max_new_tokens: int = 256, batch_size: int = 16,
             chat_template_kwargs: dict | None = None, **gen_kwargs) -> list[str]:
    """prompts: list of conversations ([{"role":..., "content":...}, ...]) or plain strings."""
    texts = [
        tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {}))
        if isinstance(p, list) else p
        for p in prompts
    ]
    gen_kwargs.setdefault("do_sample", False)  # greedy decoding = reproducible evaluation
    outputs = []
    for i in range(0, len(texts), batch_size):
        # The chat template already inserted any special tokens; don't add a second BOS.
        enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]  # drop the prompt, keep the answer
        outputs += tok.batch_decode(new_tokens, skip_special_tokens=True)
    return [o.strip() for o in outputs]
