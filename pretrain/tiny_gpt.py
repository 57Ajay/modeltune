"""Pre-train a tiny GPT from scratch on TinyStories. One file, no magic.

    python pretrain/tiny_gpt.py                                    # ~10-15 min on a Colab T4
    python pretrain/tiny_gpt.py --max_steps 300 --n_layer 2 --d_model 128 --num_stories 20000   # CPU check
    python pretrain/tiny_gpt.py --push_repo your-name/tiny-gpt-tinystories                      # + upload

This is the same objective Llama, Qwen and SmolLM are trained with: predict the next token,
minimize cross-entropy. It's just ~10,000x smaller. Chapter 02 explains every block below. Watch:
  * loss start near ln(vocab_size) (= 8.3 for 4096), the "uniform guess" baseline
  * samples go from noise -> words -> grammar -> tiny coherent stories
  * the tokens/sec and TFLOPS numbers, which connect to the compute math (6 x params x tokens)

The architecture is a miniature of a modern LLM: RMSNorm, rotary position embeddings (RoPE),
SwiGLU MLP, pre-norm residual blocks, tied input/output embeddings, PyTorch SDPA attention.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


# ------------------------------------------------------------------------------------------------
# 1. Data: train a tokenizer, turn stories into one long stream of token ids
# ------------------------------------------------------------------------------------------------
def build_data(args, out: Path):
    files = {
        "train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",  # 1 of 4 shards is plenty
        "validation": "data/validation-00000-of-00001-869c898b519ad725.parquet",
    }
    ds = load_dataset("roneneldan/TinyStories", data_files=files)
    train_text = ds["train"].select(range(min(args.num_stories, len(ds["train"]))))["text"]
    val_text = ds["validation"].select(range(min(2000, len(ds["validation"]))))["text"]

    # Byte-level BPE: start from 256 bytes, repeatedly merge the most frequent adjacent pair until
    # we have vocab_size tokens. Byte-level means no text is ever "unknown".
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_text[:50_000], trainer)
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id(EOT)

    def encode(texts):  # documents separated by <|endoftext|>, like real pre-training corpora
        ids = []
        for enc in tok.encode_batch(texts):
            ids.extend(enc.ids + [eot])
        return np.array(ids, dtype=np.uint16)

    train_ids, val_ids = encode(train_text), encode(val_text)
    print(f"tokenizer: {tok.get_vocab_size()} tokens | train {len(train_ids)/1e6:.1f}M tokens | "
          f"val {len(val_ids)/1e6:.2f}M tokens | {len(train_ids)/len(train_text):.0f} tokens/story")
    return tok, torch.from_numpy(train_ids.astype(np.int64)), torch.from_numpy(val_ids.astype(np.int64))


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    """Random windows of seq_len+1 tokens. Input = tokens[:-1], target = tokens[1:] (shifted by one)."""
    starts = torch.randint(len(data) - seq_len - 1, (batch_size,))
    chunk = torch.stack([data[s : s + seq_len + 1] for s in starts])
    return chunk[:, :-1].to(device, non_blocking=True), chunk[:, 1:].to(device, non_blocking=True)


# ------------------------------------------------------------------------------------------------
# 2. Model: a modern decoder-only transformer, small
# ------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Rescale each vector to unit RMS, then a learned gain. Cheaper than LayerNorm (no mean, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    """RoPE encodes position by ROTATING query/key pairs by an angle proportional to position, so the
    attention score q.k depends on the *relative* distance between tokens."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):  # x: (B, heads, T, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=True: token t may only attend to tokens <= t. That is what makes it a *language model*.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """The MLP used by Llama/Qwen/etc: down( silu(gate(x)) * up(x) ). This is where most params live."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(8 * d_model / 3 / 64 + 0.999) * 64  # ~2.67x width, rounded to a multiple of 64
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d_model), Attention(d_model, n_head)
        self.norm2, self.mlp = RMSNorm(d_model), SwiGLU(d_model)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying: same matrix reads and writes tokens
        cos, sin = rope_tables(seq_len, d_model // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scale residual outputs so deep stacks start stable
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:  # the ENTIRE pre-training objective is this one line
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, eot: int | None = None):
        for _ in range(max_new_tokens):  # no KV cache: simple, and fast enough at this size
            logits, _ = self(idx[:, -self.seq_len :])
            logits = logits[:, -1, :] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eot is not None and nxt.item() == eot:
                break
        return idx


# ------------------------------------------------------------------------------------------------
# 3. Training loop: AdamW, warmup + cosine LR, mixed precision, gradient clipping
# ------------------------------------------------------------------------------------------------
def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:  # warmup: Adam's statistics are garbage for the first steps
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine down to 10%


@torch.no_grad()
def evaluate(model, val, args, device, ctx, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val, args.batch_size, args.seq_len, device)
        with ctx:
            losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def sample(model, tok, device, ctx, prompt="Once upon a time", tokens=80) -> str:
    model.eval()
    idx = torch.tensor([tok.encode(prompt).ids], device=device)
    with ctx:
        out = model.generate(idx, tokens, eot=tok.token_to_id(EOT))
    model.train()
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="outputs/tiny-gpt")
    p.add_argument("--num_stories", type=int, default=300_000)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)  # small models tolerate (and need) higher LRs
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--push_repo", default=None, help="Hub repo id to upload the result to")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Same precision rule as the fine-tuning code: bf16 on Ampere+, fp16 (+ loss scaling) on T4.
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    ctx = torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32)
    scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16)

    tok, train, val = build_data(args, out)
    model = TinyGPT(tok.get_vocab_size(), args.d_model, args.n_layer, args.n_head, args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    total_tokens = tokens_per_step * args.max_steps
    print(f"model: {n_params/1e6:.1f}M params | {tokens_per_step:,} tokens/step | "
          f"{total_tokens/1e6:.0f}M tokens total = {total_tokens/n_params:.0f} tokens/param "
          f"(Chinchilla-optimal is ~20) | ~{6*n_params*total_tokens/1e15:.1f} PFLOPs of compute")

    # Weight decay on matrices only, not on norms' gains (decaying a gain toward 0 just hurts).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")

    print(f"step 0 | val loss {evaluate(model, val, args, device, ctx):.3f} "
          f"(uniform guessing = ln({tok.get_vocab_size()}) = {math.log(tok.get_vocab_size()):.2f})")
    log, t0 = [], time.time()
    for step in range(1, args.max_steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.batch_size, args.seq_len, device)
            with ctx:
                loss = model(x, y)[1] / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tame rare huge updates
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            dt = time.time() - t0
            tps = tokens_per_step * args.eval_every / dt
            val_loss = evaluate(model, val, args, device, ctx)
            log.append({"step": step, "train_loss": loss.item() * args.grad_accum, "val_loss": val_loss})
            print(f"step {step} | train {log[-1]['train_loss']:.3f} | val {val_loss:.3f} | "
                  f"lr {lr_at(step, args):.2e} | grad norm {grad_norm:.2f} | {tps:,.0f} tok/s "
                  f"(~{6 * n_params * tps / 1e12:.1f} TFLOPS)")
            print("   sample:", sample(model, tok, device, ctx).replace("\n", " ")[:300])
            t0 = time.time()

    config = {k: getattr(args, k) for k in ("vocab_size", "d_model", "n_layer", "n_head", "seq_len")}
    torch.save({"model": model.state_dict(), "config": config}, out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved to {out}")
    if args.push_repo:
        from huggingface_hub import HfApi

        (out / "README.md").write_text(
            f"# Tiny GPT pre-trained on TinyStories\n\n{n_params/1e6:.1f}M parameters, trained from scratch "
            f"for {args.max_steps} steps ({total_tokens/1e6:.0f}M tokens) with "
            "[modeltune/pretrain/tiny_gpt.py](https://github.com/57Ajay/modeltune). Final val loss: "
            f"{log[-1]['val_loss']:.3f}.\n\nArchitecture: RMSNorm, RoPE, SwiGLU, tied embeddings. "
            "Load with `TinyGPT(**config)` from that file.\n\nSample:\n\n> " + sample(model, tok, device, ctx) + "\n"
        )
        api = HfApi()
        api.create_repo(args.push_repo, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_repo)
        print(f"pushed to https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
