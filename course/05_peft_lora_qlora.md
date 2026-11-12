# 05 · PEFT: LoRA & QLoRA, fine-tuning by training almost nothing

> **Why this chapter:** chapter 04 showed full fine-tuning of a 3B model needs ~49 GB. LoRA trains ~1–2% of the
> parameters, fits on a T4, produces a file of tens of MB, and often matches full fine-tuning quality. QLoRA
> additionally stores the frozen model in 4 bits. These two ideas are why fine-tuning is accessible at all.

## 1. The idea: the *change* is low-rank

Fine-tuning changes a weight matrix `W₀ ∈ R^(d×k)` to `W₀ + ΔW`. Aghajanyan et al. (2020) showed fine-tuning
has a low *intrinsic dimension*: you can reach most of the quality while optimizing in a tiny random subspace.
**LoRA** (Hu et al., 2021) bets that ΔW is approximately **low-rank** and parameterizes it as a product:

```
h = W₀ x  +  (α / r) · B A x          A ∈ R^(r×k),  B ∈ R^(d×r),  r ≪ min(d, k)
     ▲ frozen           ▲ trained
```

- **Parameters:** `r × (d + k)` instead of `d × k`. For a 1024×3072 MLP matrix at r=16: 65,536 vs 3,145,728 (2%).
- **Initialization:** A is random, **B = 0**, so ΔW = 0 at step 0. The fine-tuned model *starts as exactly the
  original model* and moves away gradually. (This is also why DPO with a LoRA adapter starts at loss ln 2,
  chapter 10.)
- **Scaling α/r:** the update is multiplied by α/r. If you raise r but keep α fixed, each direction's effective
  learning rate shrinks. Common choices: α = r or α = 2r. **rsLoRA** (Kalajdzievski, 2023) argues for α/√r so
  high ranks still learn (`use_rslora=True` in PEFT).
- **No inference cost after merging:** `W = W₀ + (α/r)·BA` is an ordinary matrix (chapter 14).

### Worked example: Qwen3-0.6B, r = 16, all linear layers (Project 1)

| Module | Shape (out × in) | LoRA params r(d+k) |
|---|---|---|
| q_proj | 2048 × 1024 | 49,152 |
| k_proj, v_proj | 1024 × 1024 | 32,768 each |
| o_proj | 1024 × 2048 | 49,152 |
| gate_proj, up_proj | 3072 × 1024 | 65,536 each |
| down_proj | 1024 × 3072 | 65,536 |
| **per layer** | | **360,448** |
| **× 28 layers** | | **≈ 10.1M (1.7% of 596M)** |

`modeltune.train` prints this as `trainable params: …`. Check that it matches your arithmetic.

## 2. The knobs, and how to set them

| Knob | Typical | Intuition |
|---|---|---|
| `r` (rank) | 8–64 | capacity of the change. Narrow task (Project 1): 8–16. New broad behavior (Project 2, chat): 32–64 |
| `alpha` | r or 2r | update scale; tune LR first, keep α/r fixed while comparing ranks |
| `target_modules` | `all-linear` | attention-only (`q_proj, v_proj` in the original paper) is clearly weaker; MLP layers matter most |
| `dropout` | 0–0.1 | mild regularization for small datasets |
| learning rate | **1e-4 – 3e-4** for SFT | ~10× higher than full fine-tuning (1e-5 – 2e-5) because the adapter starts at zero and is scaled |

**"LoRA Without Regret"** (Thinking Machines, 2025) systematically compared LoRA with full fine-tuning:
- With LoRA on **all layers, especially the MLPs**, LoRA matched full fine-tuning on typical post-training dataset sizes.
- The optimal LoRA LR was consistently about **10× the full-FT LR**, roughly independent of rank.
- LoRA underperforms when the dataset is so big that it exceeds the adapter's capacity (pre-training-sized data).
- For **RL**, even rank 1 matched full fine-tuning, because RL delivers few bits of information per episode.

And **"LoRA Learns Less and Forgets Less"** (Biderman et al., 2024): on large continued-pre-training-style jobs
(code, math), full fine-tuning learned more, but LoRA preserved the base model's other abilities better. That's a
feature when you want a specialist that's still a decent generalist.

## 3. Why gradients still flow (and what's cheap)

The base weights are frozen, but backprop still has to pass gradients *through* them to reach earlier layers'
adapters. So LoRA does **not** save compute in the forward or backward pass through the base. It saves the
**weight gradients and optimizer states** for 98% of parameters. Memory drops massively; speed per step is only
modestly better than full FT.

## 4. QLoRA: a 4-bit frozen base

QLoRA (Dettmers et al., 2023) fine-tuned a 65B model on a single 48 GB GPU, matching 16-bit fine-tuning quality on
their benchmarks. Three ingredients:

1. **NF4 (4-bit NormalFloat):** pre-trained weights are roughly normally distributed, so NF4's 16 levels are
   placed at the **quantiles of a normal distribution**: each level is used equally often, which is
   information-theoretically optimal for such data. Weights are quantized in **blocks of 64** with one absmax
   scale per block.
2. **Double quantization:** the per-block scales themselves are quantized to 8-bit (blocks of 256), cutting their
   overhead from 0.5 to ~0.127 bits/param. Total ≈ **4.13 bits/param** ≈ 0.52 bytes.
3. **Paged optimizers:** optimizer states can spill to CPU RAM during memory spikes instead of crashing
   (`optim: paged_adamw_8bit`).

During each forward/backward, a 4-bit block is **dequantized on the fly** to the compute dtype (fp16 on T4,
bf16 on Ampere: `bnb_4bit_compute_dtype`), multiplied, and discarded. The LoRA adapter stays in 16/32-bit and is
the only thing trained. In our code that's the whole of `modeling._quantization("4bit")`:

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=hw.dtype)
```

**Costs of QLoRA:** each step is slower (the dequantization work), the base has small quantization errors
(the adapter learns to partly compensate), and you should **merge into the original 16-bit base**, not the
4-bit one (chapter 14). bitsandbytes quantizes `nn.Linear` layers only: embeddings, norms and the LM head stay
16-bit, and so do MoE experts stored as fused tensors (chapter 13).

**When to use which:**

| Situation | Choice |
|---|---|
| model × 2 bytes ≪ GPU memory (Qwen3-0.6B on T4) | **LoRA on 16-bit**: faster, no quantization error |
| model × 2 bytes ≈ or > GPU memory (3B+ on T4, 8B on 24 GB) | **QLoRA** |
| plenty of memory, big dataset, maximum quality | full fine-tuning (or high-rank LoRA) |

## 5. Variants you'll meet (all in PEFT, mostly one flag)

| Variant | Idea | Flag |
|---|---|---|
| **DoRA** (Liu et al., 2024) | split W into magnitude × direction; LoRA updates the direction, magnitude learned separately; often a bit better at low rank | `use_dora=True` |
| **rsLoRA** | scale by α/√r instead of α/r | `use_rslora=True` |
| **LoRA+** | higher LR for B than A | optimizer setting |
| **PiSSA / OLoRA** | initialize A,B from the top singular vectors of W instead of zero | `init_lora_weights="pissa"` / `"olora"` |
| **LoftQ** | initialization that compensates quantization error | `init_lora_weights="loftq"` |
| AdaLoRA, VeRA, IA³, prompt/prefix tuning | adaptive ranks, shared random matrices, vector scaling, learned soft prompts | other PEFT configs |

Start with plain LoRA on all-linear. Reach for variants only after you have a baseline and a reason.

## 6. Special cases

- **New tokens** (e.g. adding `<tool_call>`): their embedding rows are untrained, and LoRA on linear layers
  doesn't touch embeddings. Either train specific rows (`trainable_token_indices` in LoraConfig) or the whole
  embedding/LM head (`modules_to_save=["embed_tokens", "lm_head"]`, which is expensive with big vocabs). Better
  still, reuse tokens the model already knows (Project 2's template uses SmolLM3's existing `<|im_start|>` /
  `<|im_end|>`).
- **MoE experts** stored as raw `nn.Parameter` tensors aren't `nn.Linear`, so target them with
  `target_parameters` (chapter 13).
- **Multiple adapters:** one base, many adapters, swapped at runtime (`model.load_adapter`, `set_adapter`). TRL's DPO
  uses exactly this to keep a frozen reference copy of your SFT adapter (chapter 11).

## 7. In our code

1. `modeltune/modeling.py:lora_config` builds a `LoraConfig` from the YAML `lora:` block (or returns None for
   full fine-tuning / when continuing an existing adapter).
2. TRL's trainer wraps the model with `get_peft_model` and enables input gradients for checkpointing.
3. `trainer.save_model()` writes only `adapter_model.safetensors` + `adapter_config.json` (MBs, not GBs).
4. `PeftModel.from_pretrained(base, adapter)` loads it back (`evaluate`, `chat`); `merge_and_unload()` bakes it in (`merge`).

## Check yourself
1. Why must B (not A) be initialized to zero? What would happen if both were random?
2. Compute the LoRA parameter count for SmolLM3-3B at r=32 on all linear layers (36 layers; q/o 2048×2048,
   k/v 512×2048, gate/up 11008×2048, down 2048×11008).
3. You raise r from 16 to 64 with α fixed at 32 and quality drops. Why might that be?
4. Why is QLoRA slower per step than LoRA on a 16-bit base, even though it uses less memory?
5. When would you choose full fine-tuning over LoRA, even with enough memory for both?

**Next:** [06 · Data & chat templates](06_data_and_chat_templates.md)
