# 13 · Mixture of Experts: what they are, why they win, and how fine-tuning them differs

> **Why this chapter:** most frontier models are now Mixture-of-Experts (MoE): DeepSeek-V3, Qwen3-235B, Llama 4,
> gpt-oss, Kimi K2, Mixtral. They break the rule "every parameter is used for every token", and that changes
> memory math, serving, quantization and fine-tuning. Then you'll open one up and fine-tune it on Project 1's task.
>
> Notebook: [`notebooks/04_bonus_moe_granite.ipynb`](../notebooks/04_bonus_moe_granite.ipynb) ·
> Config: [`configs/bonus_moe_granite_ner_json_lora.yaml`](../configs/bonus_moe_granite_ner_json_lora.yaml)

## 1. The idea: many MLPs, each token uses a few

In a dense transformer every token goes through the same MLP in each layer (chapter 01: the MLP holds most
parameters). An MoE layer replaces that MLP with **E expert MLPs** and a **router**:

```
                       ┌─ expert 1 (SwiGLU MLP)
x ─► router: scores ───┼─ expert 2          output = Σ_{i ∈ top-k}  g_i(x) · Expert_i(x)
     s = softmax(x·W_r)│   ...
     keep top-k        └─ expert E          (the other E−k experts are skipped for this token)
```

- The **router** is a single linear layer `W_r` (hidden → E) followed by softmax/top-k. Some models renormalize
  the k gate weights to sum to 1.
- **Attention stays dense**; only the MLP is sparse (in almost all current designs).
- Each token picks its own experts *in every layer*, so different tokens of one sentence use different experts.

**Total vs active parameters:** compute per token scales with the **active** parameters (attention + k experts),
while capacity (and memory!) scales with the **total**.

### Worked example: our Granite-3.1-3B-A800M
32 layers, hidden 1536, **40 experts per layer, top-8**, each expert a SwiGLU with intermediate size 512:
- per expert: gate+up (1536 → 2×512) + down (512 → 1536) = 2.36M params; × 40 experts × 32 layers = **3.02B in experts**
- non-expert (attention, embeddings, norms, routers): ~0.28B; total **3.30B**
- active per token: 0.28B + 3.02B × 8/40 ≈ **0.88B** (hence "A800M")

So it computes like a ~0.9B dense model but has 3.3B parameters of capacity, and it needs 3.3B parameters of memory.

### The landscape (as published)

| Model | Total / active | Experts per layer (routed + shared), top-k |
|---|---|---|
| Mixtral 8x7B (2023) | 46.7B / 12.9B | 8, top-2 |
| OLMoE-1B-7B (2024, fully open) | 6.9B / 1.3B | 64, top-8 |
| DeepSeek-V3 (2024) | 671B / 37B | 256 routed + 1 shared, top-8 |
| Qwen3-30B-A3B (2025) | 30.5B / 3.3B | 128, top-8 |
| Llama 4 Maverick (2025) | 400B / 17B | 128 routed + 1 shared |
| gpt-oss-20b / 120b (2025) | 21B / 3.6B · 117B / 5.1B | 32 / 128, top-4 |
| Kimi K2 (2025) | 1T / 32B | 384 routed + 1 shared, top-8 |
| Granite-3.1-3B-A800M | 3.3B / ~0.8B | 40, top-8 |

## 2. Why MoE wins

For a fixed **training compute** budget (FLOPs ∝ active params × tokens, chapter 02), an MoE reaches a lower loss
than a dense model: the Switch Transformer (2021) reported up to ~7× faster pre-training than a dense T5 at equal
compute. At inference, cost per token tracks active params, so you get "big-model quality at small-model speed",
*if* you can afford the memory.

A short history: adaptive mixtures of experts (Jacobs et al., 1991) → sparsely-gated MoE in LSTMs (Shazeer et
al., 2017) → **GShard** (2020, top-2, translation) → **Switch Transformer** (2021, top-1, capacity factor) → **ST-MoE**
(2022, stability and fine-tuning study) → **Mixtral** (2023, open weights) → **DeepSeekMoE** (2024, fine-grained +
shared experts) → **DeepSeek-V3** (aux-loss-free balancing) → today's open frontier MoEs.

## 3. The central problem: load balancing

Left alone, routing **collapses**: an expert that's slightly better early gets chosen more, gets more gradient,
gets better, gets chosen even more. Soon a few experts do everything and the rest are dead weight. It's also an
efficiency problem: in distributed training each expert sits on some GPU, and an overloaded expert stalls the step.

Solutions:
1. **Auxiliary load-balancing loss** (Switch/GShard-style, what Hugging Face models implement as
   `load_balancing_loss_func`):
   ```
   L_aux = α · E · Σ_i  f_i · P_i      f_i = fraction of tokens routed to expert i
                                       P_i = mean router probability assigned to expert i
   ```
   It's minimized when both are uniform (1/E). Typical α ≈ 0.01: big enough to prevent collapse, small enough not
   to override the language-modeling loss. Config knobs: `router_aux_loss_coef`, `output_router_logits`.
2. **Router z-loss** (ST-MoE): penalize large router logits (`(log Σ exp(logits))²`) for numerical stability.
3. **Capacity factor / token dropping:** each expert processes at most C × (tokens/E) tokens per batch; overflow
   tokens skip the layer (only the residual passes through). "Dropless" implementations (MegaBlocks) avoid drops.
4. **Aux-loss-free balancing** (DeepSeek-V3): add a per-expert bias to the routing scores (used only for
   *selection*), nudged down for overloaded and up for underloaded experts. This balances without a loss term
   pulling against quality.
5. **Global-batch balancing** (Qwen3-MoE): compute f_i over the whole global batch instead of each micro-batch,
   so experts may specialize by domain while staying balanced overall.

## 4. Design choices: fine-grained and shared experts

- **Fine-grained experts** (DeepSeekMoE, OLMoE, Qwen3): instead of 8 big experts with top-2, use 64–256 small ones
  with top-8. Same compute, but vastly more *combinations*, and in practice better specialization.
- **Shared experts** (DeepSeek, Llama 4, Kimi K2): one or more experts that **every** token always uses, so
  common knowledge lives there and routed experts are free to specialize. (Qwen3-MoE dropped shared experts.)

## 5. What do experts actually learn?

Less than the name suggests. Mixtral's analysis found **no clear topic specialization** (biology vs code vs math
routed similarly). Routing correlated more with *syntax and token type*: for example, indentation tokens in code
going to the same expert, and consecutive tokens often sharing experts. Fine-grained designs such as OLMoE show
more domain specialization. Treat experts as a learned way to *partition computation*, not as a committee of
human-style specialists. Notebook 04 lets you look at routing yourself.

## 6. MoE at inference

- **Memory = total params.** All experts must be loaded, even though each token uses a few. A 30B-A3B model needs
  30B worth of memory.
- **Speed:** at batch size 1, decoding is memory-bandwidth bound and reads only the active experts, so it's fast.
  At large batch sizes nearly every expert is used by *some* token in each step, so the advantage shrinks to
  compute only.
- **Expert parallelism:** spread experts across GPUs; tokens are exchanged with all-to-all communication.
- **Offloading:** keep experts in CPU RAM and attention on GPU (e.g. llama.cpp's `--n-cpu-moe`). This is how people
  run 100B+ MoEs on one consumer GPU.
- **Quantization** matters even more for MoEs because their size is all in the experts. gpt-oss ships its experts
  in MXFP4 (4-bit) for exactly this reason.

## 7. Fine-tuning an MoE: what's different

**a. Memory follows *total* parameters.** Fine-tuning Granite-3B-A800M costs like a 3.3B model (6.6 GB of fp16
weights), not like a 0.8B one. Compute per token is low, so steps are relatively fast.

**b. The QLoRA trap in transformers v5.** In current transformers, each layer's experts are stored as **fused 3-D
tensors** rather than 40 separate `nn.Linear` modules:
```
block_sparse_moe.experts.gate_up_proj : [40, 1024, 1536]     (experts, out, in)
block_sparse_moe.experts.down_proj    : [40, 1536, 512]
```
That layout enables fast grouped-GEMM kernels, but **bitsandbytes only quantizes `nn.Linear`**. With
`quantization: 4bit`, attention gets quantized while the experts (≈90% of the model) **stay 16-bit**. We verified
this in the library source. On OLMoE-1B-7B the experts are 6.4B of 6.9B params, so "QLoRA" would barely shrink it
and it wouldn't fit a T4. That's why Bonus B uses a *small* MoE in plain 16-bit. (Notebook 04's last experiment lets
you measure it.)

**c. Where to put LoRA.** Three options:

| Target | Config | Trade-off |
|---|---|---|
| attention only | `target_modules: [q_proj, k_proj, v_proj, o_proj]` | cheap, stable; misses where most knowledge lives |
| **+ experts** | `target_parameters: [block_sparse_moe.experts.gate_up_proj, ...down_proj]` | PEFT's `target_parameters` (needed because experts are raw `nn.Parameter`s, not modules) adds LoRA to every expert; more capacity |
| router | (we leave it frozen) | training the router on small data can unbalance or collapse routing; usually not worth it |

"LoRA Without Regret" (chapter 05) found that LoRA on the MLP/expert layers matters most for matching full
fine-tuning, so we include the experts. Exercise: drop `target_parameters` and compare.

**d. Keep load balancing on.** Granite's config ships with `router_aux_loss_coef: 0.0`, so nothing balances the
router during your fine-tune. We set `train.router_aux_loss_coef: 0.01`; TRL then enables
`output_router_logits`, adds `0.01 × L_aux` to the loss, and logs `aux_loss`. (While building this course we hit a
TRL bug where its aux-loss code reads `config.num_experts`, but Granite/Mixtral name it `num_local_experts`.
`modeltune.modeling.load_model` patches it with one line. That kind of breakage is common at the frontier of the stack.)

**e. Instability and overfitting.** The ST-MoE study found that sparse models **overfit small fine-tuning sets more
readily** than dense ones and prefer different hyperparameters (they benefited from more regularization such as
dropout within experts, and different batch size and LR than dense models). It also found that updating only the
non-expert parameters worked surprisingly well, while updating only expert parameters hurt. Practical rules:
evaluate often, use fewer epochs, and compare attention-only vs attention+experts LoRA on *your* data.

**f. Expert-specialized fine-tuning (ESFT).** DeepSeek (2024) measured which experts a task's tokens route to and
fine-tuned *only those*: competitive quality, far fewer trainable parameters, and less forgetting. It shows that
routing statistics are a useful tool.

**g. RL on MoEs** has an extra failure mode: routing can differ between the generation pass and the training pass,
so importance ratios get noisy. That was one motivation for sequence-level ratios (GSPO, chapter 12).

## 8. Hands-on: notebook 04

1. **Count parameters:** total, in experts, and active per token (§1 math, computed live).
2. **Watch the router:** run a sentence with `output_router_logits=True` and see which of the 40 experts each layer
   picks for its tokens. Usage is spread out, without one expert for "finance" and one for "football".
3. **Baseline → fine-tune → evaluate** on exactly Project 1's data, prompt and metric.
4. **Compare with dense Qwen3-0.6B (P1):** similar active compute (~0.6B vs ~0.9B), very different total size.
   Which wins on F1? Which trained faster? Which used more memory?

## 9. Dense vs MoE cheat sheet

| | Dense | MoE |
|---|---|---|
| params used per token | all | attention + k of E experts |
| memory (inference & fine-tune) | = params | = **total** params |
| quality at equal training compute | baseline | better |
| batch-1 decode speed | ∝ params | ∝ active params (fast) |
| quantization in HF + bitsandbytes | everything in nn.Linear | experts often fused, **not** quantized by bnb |
| LoRA targets | `all-linear` | attention via `target_modules`, experts via `target_parameters` |
| extra training term | none | load-balancing aux loss (keep it on) |
| fine-tuning risks | forgetting | + routing imbalance, overfitting small data |

## Check yourself
1. Compute total vs active parameters for an MoE with 48 layers, hidden 2048, 128 experts (intermediate 768, SwiGLU), top-8, ignoring attention.
2. Why does an MoE need its load-balancing loss *during fine-tuning* too?
3. Why does `quantization: 4bit` save little memory on MoE models in transformers v5 + bitsandbytes?
4. When would you choose attention-only LoRA on an MoE?
5. At batch size 1 an MoE decodes fast; at batch size 256 the advantage shrinks. Explain.

**Next:** [14 · Shipping](14_shipping.md)
