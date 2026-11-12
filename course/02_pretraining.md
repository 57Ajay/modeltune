# 02 · Pre-training: where the model's abilities come from

> **Why this chapter:** fine-tuning is a small edit on top of an enormous pre-training run. To make good edits
> you need to know what the run produced, what it cost, and why the objective is the same one you'll use for SFT.
> Then you'll **pre-train a tiny GPT yourself** (`pretrain/tiny_gpt.py`, notebook 00).

## 1. The objective: predict the next token

Given a token sequence `x₁ … x_T`, the model outputs at every position *t* a distribution `p(· | x₁…x_t)`.
The loss is the average **cross-entropy** of the true next token:

```
L = − (1/T) Σ_t  log p(x_{t+1} | x₁ … x_t)          (nats per token)
```

Three things make this so effective:

1. **Self-supervised:** the labels are just the text shifted by one, so any text is training data.
2. **Dense signal:** with the causal mask, one forward pass trains **every position at once**. A 4k-token
   sequence is 4,096 training examples. (This is *teacher forcing*: each position conditions on the true
   previous tokens, not the model's own guesses.)
3. **It forces compression of the world:** to predict the next token of physics papers, code, dialogues and
   recipes well, the model has to model grammar, facts, reasoning patterns and style.

Useful reference points:
- Uniform guessing over a vocabulary of V tokens gives `L = ln V` (4096 tokens → 8.32). Your TinyGPT should
  start there; if it doesn't, the initialization or loss is broken.
- **Perplexity** = `exp(L)` = "as uncertain as choosing among this many tokens".
- Loss is only comparable **with the same tokenizer and data**. Bigger tokens mean higher loss per token for the
  same text.

**Connection to fine-tuning:** SFT uses exactly this loss. The only differences are the data (demonstrations
instead of web text) and a **mask** that excludes the prompt from the loss (chapter 06). Continued pre-training
is literally this chapter on domain text.

## 2. Data: the real secret sauce

Frontier-scale pre-training corpora are tens of trillions of tokens: Llama 3 (2024) used about 15T, SmolLM3 (2025)
11.2T, Qwen3 (2025) about 36T. The pipeline:

| Stage | What | Why |
|---|---|---|
| Collect | Common Crawl web dumps, code (GitHub), papers, books, math, multilingual | coverage |
| Extract | HTML → clean text | boilerplate is noise |
| Filter | language ID, heuristics (length, symbol ratio, repetition), **quality classifiers** | most of the web is low quality |
| Deduplicate | exact + fuzzy (MinHash) | duplicates waste compute and cause memorization |
| Decontaminate | remove benchmark test sets | otherwise evals measure memory, not skill |
| Mix | choose proportions (web/code/math/…) and change them over training | the mixture shapes the model's strengths |

A landmark result: **FineWeb-Edu** (HuggingFace, 2024) used an LLM to score web pages for "educational value"
and kept only the best ~1.3T of 15T tokens; models trained on it beat models trained on far more unfiltered
text. **Quality beats quantity.** You'll see the same lesson in fine-tuning (LIMA, chapter 06).

Late in training, labs **anneal** on the highest-quality data (math, code, curated text) with a decaying
learning rate, often called *mid-training*, and extend context length (e.g. 4k → 32k+ with RoPE scaling).
By the time you download a "base" model, it has already seen a lot of instruction-like data.

## 3. Tokenizer training

Before the model, the tokenizer is trained on a data sample with **byte-level BPE**: start with 256 byte
tokens, repeatedly merge the most frequent adjacent pair into a new token, until reaching the vocab size.
Byte-level means nothing is ever out-of-vocabulary. Trade-off:

- **Bigger vocab** (Qwen3: 151,936; Llama 3: 128,256; Gemma 3: 262,144) gives fewer tokens per text (cheaper,
  longer effective context, better multilingual coverage) but bigger embedding matrices. That matters a lot for
  small models: 26% of Qwen3-0.6B is embeddings.
- The tokenizer is **frozen forever** after pre-training. Adding tokens later means new embeddings with no
  training (chapter 06).

`tiny_gpt.py` trains a 4,096-token BPE on 50k stories in a few seconds. Try `--vocab_size 512` vs `8192`.

## 4. Compute: the 6ND rule

Training FLOPs ≈ **6 × N × D**, with N = parameters and D = training tokens. The forward pass costs ~2 FLOPs per
parameter per token (a multiply and an add); backward costs ~2× forward.

Worked example, Qwen3-0.6B: 6 × 0.6·10⁹ × 36·10¹² ≈ **1.3·10²³ FLOPs**.
- One Colab T4 (65 TFLOPS fp16 peak, maybe ~25 achieved): about 5·10⁹ s, roughly **165 years**.
- 1,000 H100s (≈ 1,000 TFLOPS bf16 peak each, ~40% utilization): about 3·10⁵ s, roughly **4 days**.

So pre-training a real model is out of reach, while fine-tuning it on 10⁶–10⁷ tokens costs about 10¹⁶ FLOPs,
minutes on a T4. That gap of seven orders of magnitude is why everyone fine-tunes.

**MFU (model FLOPs utilization)** = achieved FLOPs / peak FLOPs. `tiny_gpt.py` prints achieved TFLOPS so you can
compute yours. Small models get low MFU because their matrices are too small to saturate the GPU.

## 5. Scaling laws: how big, how long?

- **Kaplan et al. (2020):** loss falls as a smooth power law in N, D and compute. Bigger is predictably better.
- **Chinchilla (Hoffmann et al., 2022):** for a fixed compute budget, the loss-optimal split is about **20
  tokens per parameter** (their 70B model on 1.4T tokens beat the 280B Gopher).
- **Modern practice "over-trains" small models** far past Chinchilla because *inference* cost depends only on N.
  Llama 3 8B saw ~1,900 tokens/param; Qwen3-0.6B ~60,000 tokens/param.

**Implication for you:** modern small models are extremely knowledge-dense for their size. Your fine-tuning data
(10⁴–10⁷ tokens) is a drop in that ocean, so it can easily *steer behavior* but will not add much *knowledge*.

## 6. Optimization recipe (shared with fine-tuning)

| Ingredient | Pre-training typical | Why |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 | per-parameter adaptive steps; decoupled decay regularizes |
| LR schedule | linear **warmup**, then cosine decay (or **WSD**: warmup–stable–decay) | early Adam statistics are noisy; decaying at the end "settles" the weights |
| Batch size | millions of tokens per step | lower gradient noise; better hardware use |
| Precision | bf16 mixed precision | 2× speed, half memory, fp32 range (chapter 04) |
| Stability | gradient clipping (1.0), QK-norm, z-loss | loss spikes can kill a run |

WSD (used by e.g. MiniCPM and SmolLM3) keeps the LR flat and decays only at the end, which makes it easy to
continue training from a mid-run checkpoint. Fine-tuning reuses the same ingredients at smaller scale: warmup,
cosine, AdamW, clipping.

## 7. What a base model is (and isn't)

A base model is a superb **document completer**. Ask it "What is the capital of France?" and it might continue
with "What is the capital of Germany? What is…", because on the web, questions often come in lists. It has:
- broad knowledge and skills (from pre-training),
- **no** notion of "user" and "assistant", no chat template, no instinct to answer and stop.

Turning that into an assistant is **post-training** (chapter 03). You'll do it yourself in Project 2.

## 8. Hands-on: TinyGPT (`pretrain/tiny_gpt.py`)

Run notebook 00 (≈10–15 min on a T4, estimate). The file is ~300 lines and mirrors this chapter:

| Code | Concept |
|---|---|
| `build_data` | BPE tokenizer training; documents joined with `<|endoftext|>` into one token stream |
| `get_batch` | random windows; **input = tokens[:-1], target = tokens[1:]** |
| `RMSNorm`, `apply_rope`, `Attention`, `SwiGLU`, `Block` | the modern architecture from chapter 01 |
| `is_causal=True` | the causal mask |
| `F.cross_entropy(...)` in `forward` | the entire objective |
| `lr_at` | warmup + cosine (to 10%) |
| AdamW with decay on matrices only | standard practice: don't decay norm gains |
| `GradScaler` | fp16 loss scaling on T4 (chapter 04) |
| `clip_grad_norm_` | stability |

Default model: d_model 384, 6 layers, 6 heads, about **12M parameters**, 16k tokens per step, 4,000 steps (~65M
tokens). What to observe:
1. **Step 0 val loss ≈ 8.3 = ln(4096).** The init is sane.
2. Loss drops fast, then slowly. Samples go from random tokens → frequent words → grammatical sentences →
   short coherent stories with names and simple plots.
3. Train and val loss stay close: there's enough data that the model doesn't memorize. Shrink `--num_stories`
   to see them diverge (overfitting).
4. The printed **tokens/param** is below Chinchilla's 20. Raise `--max_steps` if you have time and watch the
   loss keep falling.

## 9. Pre-training vs fine-tuning at a glance

| | Pre-training | Fine-tuning |
|---|---|---|
| Data | 10¹³ tokens of raw text | 10⁴–10⁸ tokens of curated examples |
| Loss | next token, all tokens | next token on *answers only* (SFT), or preference/RL losses |
| Compute | 10²²–10²⁵ FLOPs | 10¹⁵–10¹⁹ FLOPs |
| Learns | knowledge, skills, language | format, behavior, style, task focus |
| Risk | divergence, data quality | forgetting, overfitting, format bugs |

## Check yourself
1. Why is the initial loss of a randomly initialized model ≈ ln(vocab_size)? What would ln(151,936) be?
2. Estimate the FLOPs to fine-tune Qwen3-0.6B on 4,000 examples × 150 tokens for one epoch. How does it compare to its pre-training?
3. Why do labs train small models on far more than 20 tokens per parameter?
4. If you double the tokenizer vocabulary, what happens to loss-per-token and to tokens-per-document?

**Next:** [03 · The post-training landscape](03_post_training_landscape.md)
