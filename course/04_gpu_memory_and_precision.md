# 04 · GPU memory & precision: where every byte goes

> **Why this chapter:** "CUDA out of memory" is the first wall everyone hits. If you can estimate memory on paper,
> you can choose between full fine-tuning, LoRA and QLoRA *before* you start, set batch sizes sensibly, and debug
> OOMs in minutes. It also explains the T4's fp16-vs-bf16 quirk that our code handles automatically.

## 1. The four things that live in GPU memory during training

```
total ≈ weights + gradients + optimizer states + activations (+ framework overhead ~0.5–1 GB)
```

| Item | Size | Notes |
|---|---|---|
| **Weights** | params × bytes/param | 2 B for fp16/bf16, ~0.5 B for 4-bit |
| **Gradients** | trainable params × bytes | only for parameters you train |
| **Optimizer states** | AdamW: 2 numbers per trainable param (momentum, variance), usually fp32 = 8 B | 8-bit Adam: 2 B |
| **Activations** | grows with batch × sequence length × layers × hidden size | saved during forward for use in backward |

### Full fine-tuning with mixed precision: ~16 bytes per parameter
The standard accounting (from the ZeRO paper): 16-bit weights (2) + 16-bit grads (2) + fp32 master weights (4) +
Adam momentum (4) + Adam variance (4) = **16 bytes/param**, before activations.

| Model | Full FT (16 B/param) | LoRA, 16-bit base | QLoRA, 4-bit base |
|---|---|---|---|
| Qwen3-0.6B (0.60B) | ~9.5 GB | ~1.2 GB + adapter | ~0.6 GB + adapter |
| SmolLM3-3B (3.08B) | **~49 GB** ❌ T4 | ~6.2 GB + adapter | **~2 GB** + adapter |
| Llama-3.1-8B (8B) | ~128 GB ❌ | ~16 GB ❌ T4 | ~5.5 GB ✅ |

*(weights-and-states only; add activations)*

With LoRA the optimizer and gradients only exist for the adapter: Qwen3-0.6B at r=16 on all linear layers has
about 10.1M trainable params, and 10.1M × (4 weight + 4 grad + 8 Adam) ≈ **160 MB**. That's why LoRA makes the
model's own size the main cost (chapter 05).

## 2. Activations: the part that scales with your data

During the forward pass, each layer stores intermediate tensors that the backward pass needs. Roughly, for one
layer (Korthikanti et al., 2022, with a memory-efficient attention kernel):

```
activations per layer ≈ c × batch × seq_len × hidden × bytes     (c ≈ 15–20 for a SwiGLU block in 16-bit)
```

SmolLM3-3B (36 layers, hidden 2048), batch 2, seq 1024, fp16: about 36 × 17 × 2 × 1024 × 2048 × 2 B ≈ **5 GB**.
Double the sequence length and activations double. (Without memory-efficient attention there's also a
seq²-sized attention matrix per head, which is why SDPA/FlashAttention matter.)

### The logits trap
The final layer produces `batch × seq × vocab` logits, often upcast to fp32 for the loss:
2 × 1024 × 128,256 × 4 B ≈ **1 GB**, plus its gradient. Large vocabularies make this a top-3 memory item for
small models. Fused/chunked cross-entropy kernels (Liger, Unsloth; TRL computes the loss with a fused LM head in
recent versions) avoid materializing it all at once.

### Gradient checkpointing: trade compute for memory
Instead of storing every intermediate, store only each layer's *input* and recompute the rest during backward:
- activations drop to about `layers × batch × seq × hidden × 2 B` (≈ 0.3 GB in the example above) plus one layer's worth,
- you pay one extra forward pass, roughly **20–35% slower**.

Turned on with `gradient_checkpointing: true` (Project 2, MoE). It's the single biggest memory lever after
batch size.

## 3. Number formats

| Format | Bits (sign·exponent·mantissa) | Range | Precision | Hardware |
|---|---|---|---|---|
| fp32 | 1·8·23 | ~10±38 | ~7 digits | everywhere |
| tf32 | 1·8·10 (matmul internal) | like fp32 | ~3 digits | Ampere+ (automatic) |
| **fp16** | 1·5·10 | **max 65,504**, tiny values underflow below ~6·10⁻⁸ | ~3 digits | Volta/Turing+ tensor cores (**T4**) |
| **bf16** | 1·8·7 | same as fp32 | ~2 digits | **Ampere+** (A100, L4, RTX 30/40, H100) |
| fp8 (E4M3/E5M2) | 8 | small | low | Hopper+ (H100), Blackwell |
| int8 | 8 | integer + scale | | weight quantization |
| **NF4** | 4 | quantile-based + per-block scale | | bitsandbytes (QLoRA) |
| MXFP4 | 4 (+ shared 8-bit scale per 32 values) | | | gpt-oss weights; native on Blackwell |

**bf16 vs fp16:** same 16 bits, different trade. bf16 keeps fp32's 8 exponent bits (huge range, coarse
precision); fp16 has more mantissa (finer precision) but overflows above 65,504 and underflows tiny gradients.
Models are pre-trained in bf16 today, so their activations sometimes exceed fp16's range. That's a real risk on a
T4. (We checked our three models: max activations of ~400 for SmolLM3, ~6.6k for Qwen3-0.6B and ~10k for
Granite-MoE on sample text, all inside fp16's range. Some models, notably the Gemma family, are known to overflow
in fp16.)

### Mixed precision and loss scaling
Mixed precision = do the matmuls in 16-bit (fast tensor cores), keep a **fp32 master copy** of trained weights
for the update (tiny updates like 1e-4 × 1e-3 would vanish in 16-bit).

With **fp16** you also need **loss scaling**: multiply the loss by a large S (e.g. 65,536) so small gradients
don't underflow to zero, unscale before the optimizer step, and if any gradient became inf/NaN, skip the step
and halve S (PyTorch `GradScaler`, which the HF Trainer uses when `fp16=True`). bf16 doesn't need it.

### The T4 story (what `modeltune/hardware.py` does)
The Colab T4 is a Turing GPU (compute capability 7.5): 16 GB, ~65 TFLOPS fp16 tensor-core peak, **no native
bf16**, and no FlashAttention-2 (which needs Ampere, 8.0+). So:

```python
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
```
- T4/V100 → fp16 + loss scaling. Ampere+ → bf16. CPU → fp32.
- We don't use `torch.cuda.is_bf16_supported()` because it can return True on a T4 (bf16 *emulation*), which is slow.
- With **full** fine-tuning on fp16 hardware we load the model in fp32 (master weights) and let autocast run
  matmuls in fp16; the GradScaler refuses to unscale fp16 parameters. With LoRA, PEFT keeps the small adapter in
  fp32 automatically, so the frozen base can stay 16-bit.
- Attention uses PyTorch **SDPA**, which picks the best kernel available on any GPU.

## 4. Throughput and memory levers (in the order you should reach for them)

| Lever | Config | Memory | Speed | Notes |
|---|---|---|---|---|
| Smaller per-device batch + **gradient accumulation** | `per_device_train_batch_size`, `gradient_accumulation_steps` | ↓↓ | ≈ | *effective batch* = per_device × accum × GPUs stays the same |
| Shorter sequences | `max_length` | ↓↓ | ↑ | check your length distribution first; don't truncate answers |
| Gradient checkpointing | `gradient_checkpointing: true` | ↓↓↓ | ↓ 20–35% | |
| 8-bit / paged optimizer | `optim: paged_adamw_8bit` | ↓ (states 8→2 B/param) | ≈ | "paged" spills to CPU RAM on spikes instead of OOM-ing |
| QLoRA | `model.quantization: 4bit` | ↓↓↓ (weights ÷4) | ↓ (dequantize on the fly) | chapter 05 |
| Packing | `packing: true` | – | ↑↑ | concatenates short examples into full-length rows (less padding); best with FlashAttention so examples don't attend to each other |
| Fused kernels | Liger kernel, Unsloth | ↓ | ↑ | Unsloth is popular on T4: roughly 2× speed with custom kernels |

**OOM debugging order:** halve batch (double accumulation) → turn on checkpointing → lower `max_length` → 8-bit
optimizer → 4-bit base → smaller model. After every change, read the printed **peak GPU memory** (`modeltune.train`
prints it) and compare it with the 15 GB a Colab T4 actually gives you.

## 5. Worked estimate: Project 2 on a T4

SmolLM3-3B, QLoRA r=32 all-linear, batch 2, seq 1024, checkpointing on:
- 4-bit weights ≈ 1.5 GB (+ embeddings kept in 16-bit ≈ 0.5 GB) → **~2 GB**
- LoRA r=32 on all linear layers ≈ 60M params × 16 B (fp32 weight + grad + Adam) → **~1 GB** (8-bit Adam: less)
- activations with checkpointing ≈ 0.5–1 GB; logits + grad ≈ 2 GB → **~3 GB**
- CUDA context, buffers, fragmentation → **~1 GB**

≈ **7 GB**, comfortably under 15 GB. Without QLoRA (16-bit base) it'd be about 11 GB, possible but tight;
without checkpointing, about +5 GB. Do this arithmetic before every project.

## 6. Scaling beyond one GPU (for later)

| Strategy | What's split | Memory per GPU | Use when |
|---|---|---|---|
| **DDP** (data parallel) | data | full copy of everything | model fits one GPU; want speed |
| **FSDP / DeepSpeed ZeRO-1/2/3** | optimizer states / + grads / + weights | ÷ N GPUs | model doesn't fit one GPU |
| Tensor parallel | each matmul | ÷ N | very large models, fast interconnect |
| Pipeline parallel | layers | ÷ N | very large models, many nodes |
| Expert parallel | MoE experts | ÷ N | MoE models (chapter 13) |

Our code is DDP-ready: `accelerate launch -m modeltune.train CONFIG` runs one process per GPU (each loads the
model on its own `LOCAL_RANK`). For FSDP/ZeRO, configure Accelerate (`accelerate config`); TRL supports both.

## Check yourself
1. Why does full fine-tuning need ~16 bytes/param but QLoRA's frozen weights only ~0.5?
2. Your run OOMs at batch 8, seq 2048. List three changes that keep the *effective* batch size at 8.
3. Why can bf16 training skip loss scaling while fp16 can't?
4. Estimate the logits memory for Qwen3-0.6B (vocab 151,936) at batch 16 × seq 512 in fp32. Surprised?
5. On a T4, why do we load the model in fp32 for full fine-tuning but fp16 for LoRA?

**Next:** [05 · PEFT: LoRA & QLoRA](05_peft_lora_qlora.md)
