# 01 · LLM refresher: what is actually inside the thing we fine-tune

> **Why this chapter:** every fine-tuning decision (which layers get LoRA, why memory explodes with context
> length, why the chat template matters) maps to a specific part of the architecture. This is a fast, practical
> tour of a modern decoder-only transformer, using **Qwen3-0.6B** (Project 1) as the running example.

## 1. The pipeline in one picture

```
"The cat sat"                                   text
   │ tokenizer (BPE)
   ▼
[785, 8251, 7578]                               token ids            (vocab = 151,936 for Qwen3)
   │ embedding lookup  E[vocab × d]
   ▼
x ∈ R^(T × d)                                    T tokens, d = 1024 numbers each
   │ 28 × transformer block:   x = x + Attention(RMSNorm(x))
   │                           x = x + MLP(RMSNorm(x))
   ▼
RMSNorm → x · Eᵀ  (unembedding, tied to E)
   ▼
logits ∈ R^(T × vocab)  → softmax → P(next token | everything before it), at EVERY position
```

Everything an LLM does, from chatting and coding to "reasoning", is this function applied repeatedly: predict a
distribution over the next token, pick one, append it, repeat.

## 2. Tokens

A tokenizer splits text into sub-word pieces from a fixed vocabulary, learned with **BPE** (byte-pair encoding):
start from bytes, repeatedly merge the most frequent adjacent pair (you'll train one in chapter 02). Consequences
you will hit when fine-tuning:

- **Cost and context are measured in tokens**, not words. English is about 0.75 words per token; code, JSON and
  non-English text often take more tokens per word.
- **Special tokens** like `<|im_start|>`, `<|im_end|>` and `<|endoftext|>` are single tokens with special meaning.
  They structure conversations and tell the model when to stop (chapter 06). Most "my model never stops
  talking" bugs are special-token bugs.
- Numbers are split in tokenizer-specific ways: Qwen splits them into single digits, Llama 3 into chunks of
  up to three ("1234" → "123" + "4"). This is part of why arithmetic is hard for LLMs.

## 3. Embeddings and the residual stream

Each token id indexes a row of the embedding matrix `E` (151,936 × 1,024 for Qwen3-0.6B = **156M parameters**,
26% of this small model!). The resulting vectors flow through the network along the **residual stream**:
every block *reads* from it and *adds* its output back (`x = x + f(x)`). That additive structure is why LoRA
works so well (small additive updates to a few matrices), and why you can stack dozens of layers stably.

**Tied embeddings:** small models often reuse `E` as the output layer (`logits = x · Eᵀ`) to save parameters.
Qwen3-0.6B and SmolLM3 both do this.

## 4. Attention: how tokens talk to each other

For each token, three linear projections produce a **query** (what am I looking for?), a **key** (what do I
contain?) and a **value** (what do I pass on?):

```
Q = x Wq,  K = x Wk,  V = x Wv
Attention(Q, K, V) = softmax( Q Kᵀ / √d_head  +  causal_mask ) V
```

- The **causal mask** sets scores for future positions to −∞, so token *t* only sees tokens ≤ *t*. That's what
  makes it a *language model* you can train on every position in parallel (chapter 02).
- **Multi-head:** the `d` dims are split into heads (Qwen3-0.6B: 16 query heads × 128 dims) that attend
  independently, then `Wo` mixes them back.
- **Cost:** the `QKᵀ` score matrix is T × T, so compute grows quadratically with sequence length. Memory-efficient
  kernels (FlashAttention, PyTorch SDPA) avoid *storing* that matrix, which is why `max_length` is the most
  important memory knob after batch size (chapter 04).

### Modern attention details you'll see in configs
- **GQA (grouped-query attention):** several query heads share one key/value head. Qwen3-0.6B has 16 Q heads but
  8 KV heads; SmolLM3-3B has 16 Q / 4 KV. This shrinks the **KV cache** (below) with little quality loss. It's why
  `k_proj` is smaller than `q_proj` in the layer listing:
  `q_proj: 1024→2048, k_proj: 1024→1024, v_proj: 1024→1024, o_proj: 2048→1024`.
- **RoPE (rotary position embeddings):** position is injected by *rotating* each query/key pair by an angle
  proportional to its position. The dot product `q·k` then depends only on the *relative* distance. RoPE has no
  learned parameters, and its base frequency can be stretched to extend context (YaRN, etc.). SmolLM3 removes
  RoPE from every 4th layer ("NoPE") to help long context.
- **QK-norm:** Qwen3 normalizes q and k per head before the dot product (stability).

## 5. The MLP: where most parameters live

After attention mixes information *across* tokens, the MLP transforms each token *independently*. Modern
models use **SwiGLU**:

```
MLP(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )        Qwen3-0.6B: 1024 → 3072 → 1024
```

Per layer, the MLP has 3 × 1024 × 3072 ≈ 9.4M params versus ≈ 6.3M for attention. MLPs hold much of a model's
"knowledge", and they are exactly what a **Mixture-of-Experts** replicates into many experts (chapter 13).

## 6. Normalization: RMSNorm, pre-norm

Each sub-layer's *input* is normalized (`pre-norm`). RMSNorm rescales the vector to unit root-mean-square and
multiplies by a learned gain, a cheaper LayerNorm with no mean subtraction and no bias. Norm weights are tiny in
count but sensitive, which is one reason QLoRA keeps them in higher precision.

## 7. The full layer list (memorize this; it's what LoRA targets)

```
model.embed_tokens                         (vocab × d)
model.layers.N.self_attn.{q,k,v,o}_proj    ← LoRA "attention" targets
model.layers.N.mlp.{gate,up,down}_proj     ← LoRA "MLP" targets
model.layers.N.{input,post_attention}_layernorm
model.norm
lm_head                                    (often tied to embed_tokens)
```

`target_modules: all-linear` in our configs = all seven projections in every layer (PEFT excludes `lm_head`).
You can verify the names for any model with `print(model)`.

## 8. Inference: logits → text

At each step the model gives logits for the next token. **Decoding** turns them into a choice:

| Strategy | What it does | Use for |
|---|---|---|
| greedy (`do_sample=False`) | always the argmax | evaluation (reproducible), extraction tasks |
| temperature *τ* | divide logits by τ before softmax: τ<1 sharper, τ>1 flatter | creativity vs safety |
| top-k / top-p | sample only from the k most likely / smallest set with cumulative prob ≥ p | cut off the garbage tail |
| repetition penalty | down-weight already-used tokens | patching loops (a symptom, not a fix) |

Generation stops when the model emits the **EOS / end-of-turn token** (or hits `max_new_tokens`). If
fine-tuning never showed the model *where* to put that token, it won't stop (chapter 06).

**KV cache:** during generation, the keys/values of past tokens are cached so each new token only computes its
own. Cache memory = `2 × layers × kv_heads × head_dim × T × bytes`. For Qwen3-0.6B at fp16 that's
2 × 28 × 8 × 128 × 2 B ≈ **115 KB per token**, about 0.9 GB for an 8k-token context. GQA directly shrinks this.

**Batched generation needs left padding:** prompts of different lengths are padded on the *left* so every
sequence's last real token sits at the same position and generation continues from it
(`modeltune/inference.py` sets `padding_side = "left"`). Training pads on the right, which is fine because there
we compute all positions at once.

## 9. Where the parameters go (Qwen3-0.6B, 596M total)

| Part | Params | Share |
|---|---|---|
| Embedding (tied with output) | 156M | 26% |
| Attention (28 layers) | 28 × 6.3M ≈ 176M | 30% |
| MLP (28 layers) | 28 × 9.4M ≈ 264M | 44% |

For bigger models the embedding share shrinks (SmolLM3-3B: 263M of 3.08B ≈ 9%) and the MLP share grows.
Holding these proportions in your head lets you estimate memory and LoRA sizes instantly (chapters 04–05).

## 10. Base vs instruct vs reasoning models

Same architecture, different training (chapter 03):
- **Base** (`-Base`, `-pt`): trained only to continue text. Knows a lot, follows instructions poorly, has no chat format.
- **Instruct / chat**: base + post-training (SFT, preference optimization, RL). Follows a chat template.
- **Reasoning / "thinking"**: instruct + RL that rewards long chains of thought before answering. Qwen3 models are
  *hybrid*: `enable_thinking=True/False` in the chat template toggles a `<think>…</think>` block. We disable it
  for Project 1 (we want JSON, immediately).

## Check yourself
1. Why is the KV cache per token smaller with GQA? Compute it for SmolLM3-3B (36 layers, 4 KV heads, head_dim 128, fp16).
2. Where would a LoRA adapter on `all-linear` attach in a layer? Why is the MLP usually the bigger target?
3. Why do we pad on the left for generation but on the right for training?
4. A model's outputs never end. Name two places in the pipeline where the bug could be.

**Next:** [02 · Pre-training](02_pretraining.md)
