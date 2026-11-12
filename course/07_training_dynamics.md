# 07 · Training dynamics: hyperparameters, loss curves, and debugging

> **Why this chapter:** with data and memory sorted, a handful of numbers decide whether training works: the
> learning rate above all. You also need to *read* a training run: what healthy and broken loss curves look like,
> and what to do about each.

## 1. Starting hyperparameters

These are the defaults that usually work. Change one at a time, and only for a reason.

| | SFT · LoRA (P1) | SFT · QLoRA (P2) | Full FT | DPO · LoRA (P2) | GRPO · LoRA |
|---|---|---|---|---|---|
| learning rate | 2e-4 | 1e-4 – 2e-4 | 1e-5 – 2e-5 | 5e-6 – 5e-5 | 1e-6 – 2e-5 |
| schedule | cosine, warmup 3–5% | cosine, warmup 3% | cosine, warmup 3% | cosine, warmup 10% | constant + short warmup |
| effective batch | 16–64 | 16–64 | 64–256 | 16–64 | 16–64 completions |
| epochs | 1–3 | 1–2 | 1–3 | 1 | – (steps) |
| weight decay | 0–0.01 | 0 | 0.0–0.1 | 0 | 0 |
| max_grad_norm | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LoRA r / α | 16 / 32 | 32 / 32–64 | – | (continue SFT adapter) | 16 / 32 |

Why these LRs differ:
- **LoRA ≈ 10× full FT.** The adapter starts at zero, is scaled by α/r, and has few parameters (chapter 05).
- **DPO ≪ SFT.** DPO starts from a good model and only *nudges* relative probabilities. High LR quickly
  "over-optimizes" (degenerate, very long or repetitive outputs).
- **GRPO lowest.** RL gradients are noisy and the model generates its own training data, so instability compounds.

## 2. The learning rate is the knob

| Symptom | Likely LR issue |
|---|---|
| loss spikes, NaN, outputs turn to gibberish, model forgets everything | too high |
| loss barely moves; outputs identical to base | too low (or a bug, see §6) |
| fast drop then plateau at a mediocre level | maybe too high late in training (decay helps) or capacity-limited |

**How to tune:** run a 3-point sweep, ×3 apart (e.g. 6e-5, 2e-4, 6e-4), on a small slice of the data, and compare
the **task metric**, not just loss. Pick the best and refine if needed.

**Warmup** (a few % of steps of linearly rising LR) avoids huge early updates while Adam's moment estimates are
still garbage. **Cosine decay** lowers the LR toward the end so the weights settle. In transformers v5,
`warmup_steps` takes a float < 1 to mean a ratio (`warmup_ratio` was removed).

## 3. Batch size, accumulation, epochs, sequence length

- **Effective batch** = `per_device_train_batch_size × gradient_accumulation_steps × num_GPUs`. Accumulation
  gives you the *statistics* of a big batch with the *memory* of a small one. When you change the effective
  batch a lot, re-tune LR (bigger batch tolerates higher LR, roughly ∝ √batch).
- **Steps** = examples × epochs / effective batch. Project 1: 4,000 × 1 / 16 = 250 steps. Make sure warmup and
  eval intervals make sense in *steps*.
- **Epochs:** each pass over the same data increases memorization. For SFT, 1–3 epochs; watch validation. For DPO,
  1 epoch is standard.
- **`max_length`** caps tokens per example (memory scales linearly with it). Set it from the data's length
  distribution, not by habit.

## 4. Reading a training run

TRL logs every `logging_steps`: `loss`, `grad_norm`, `learning_rate`, `mean_token_accuracy` (how often the
argmax prediction equals the target token, on loss tokens), `entropy`, and with an eval split `eval_loss`.

```
Healthy                 Overfitting             Not learning            Unstable
loss                    loss                    loss                    loss
 │╲                      │╲     eval              │──────────────          │╲  ╱╲   ╱
 │ ╲___ eval             │ ╲___╱                  │                        │ ╲╱  ╲_╱  NaN
 │  ╲____ train          │  ╲______ train         │                        │
 └──────── steps         └──────── steps          └──────── steps          └──────── steps
```

| Pattern | Means | Do |
|---|---|---|
| train ↓, eval ↓, then flat together | learning, converged | stop; evaluate the task metric |
| train ↓, eval ↑ | overfitting / memorizing | fewer epochs, more data, lower LR or r, more dropout; early stopping |
| loss drops in a **staircase** at each epoch boundary | memorizing specific examples | that's overfitting with multiple epochs |
| loss flat from step 0 | LR too low, all tokens masked, or the adapter isn't attached | check the tokenized-example printout and trainable params |
| loss starts *very* low | targets are trivially predictable or leaked into the prompt | inspect the data |
| spikes / NaN | LR too high, fp16 overflow, bad examples | lower LR, check `grad_norm`, try bf16 hardware, filter data |
| `grad_norm` creeping up | instability building | lower LR; `max_grad_norm` clips the damage but doesn't fix the cause |

**Calibrating absolute values:** for SFT on chat data an instruct model often starts around 1–2 nats/token and
ends around 0.6–1.2. A rigidly formatted task like Project 1's JSON can go much lower (most tokens are
deterministic braces and key names). Low loss is not the goal. **Eval loss is not quality either**: LIMA's eval
loss went *up* over epochs while human-judged quality kept improving. Trust the task metric (chapter 08).

## 5. Regularization and forgetting

- **Early stopping:** `save_strategy: steps`, `eval_strategy: steps`, `load_best_model_at_end: true`,
  `metric_for_best_model: eval_loss`, `save_total_limit: 2`.
- **LoRA dropout** (0.05–0.1) and **weight decay** help a little on small datasets.
- **NEFTune** (`neftune_noise_alpha: 5`): adds noise to input embeddings during SFT; Jain et al. (2023) reported
  large gains on chat benchmarks. It's cheap to try for chat SFT, though its value for narrow tasks is unclear.
- **Forgetting:** mix in general data, lower the LR, fewer epochs, LoRA over full FT, and *measure* general
  ability before/after (chapter 08).

## 6. The debugging checklist (run it top to bottom when something's off)

1. **Read the printed tokenized example.** Is the template right? Is the loss on the answer only? Does it end with
   the EOS/end-of-turn token?
2. **Check `trainable params`.** 0 or suspiciously small means the adapter targets matched nothing (wrong module
   names for this architecture: `print(model)`).
3. **Overfit a tiny batch:** `data.max_train_samples=16 train.num_train_epochs=50`. Loss should go near zero and
   the model should reproduce those 16 answers. If it can't, the pipeline is broken: data, masking, LR, or frozen
   weights.
4. **Generate from the trained adapter** on training examples. Does `evaluate` actually load the adapter? (It
   prints `adapter=...`.)
5. **Precision:** NaNs only on a T4 but not on bf16 hardware means fp16 overflow. Lower LR, or check activation
   magnitudes as in chapter 04.
6. **Compare with the base** on the same prompts (`--base`). If they're identical, the adapter isn't being used or
   didn't learn.

## 7. Reproducibility and noise

- Set `seed` (our configs do). Results still vary run to run, especially with small data.
- If two configs differ by less than the run-to-run spread, the difference isn't real. Run 2–3 seeds before
  concluding.
- Log everything: `train.report_to: trackio` (or `wandb`, `tensorboard`) gives you curves; the resolved config is
  saved to `outputs/<run>/modeltune_config.yaml`.

## 8. Surviving Colab

Free Colab sessions disconnect. For long runs: `train.save_strategy=steps train.save_steps=100`, keep
`hub.push=true` so the final adapter lands on the Hub, and resume with
`train.resume_from_checkpoint=true` (`modeltune.train` passes it through to `trainer.train`). Copy
`outputs/` to Google Drive if you need checkpoints to survive a disconnect.

## Check yourself
1. Train loss ↓, eval loss ↑ after epoch 1, but your task F1 is still improving at epoch 2. What do you do and why?
2. Effective batch 16 at LR 2e-4 works. You move to an A100 and use batch 128. What LR do you try first?
3. Loss is flat at the first step and stays there; trainable params is 10M. Name two causes.
4. Why are DPO learning rates so much lower than SFT ones?

**Next:** [08 · Evaluation](08_evaluation.md)
