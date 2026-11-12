# 06 · Data & chat templates: the part that actually decides the outcome

> **Why this chapter:** the model learns *exactly* what the token sequence shows it, nothing more. Most failed
> fine-tunes are data bugs: wrong template, missing end-of-turn token, loss on the wrong tokens, inconsistent
> labels. This chapter is the checklist that prevents them.

## 1. The standard formats

TRL (and therefore `modeltune`) understands a handful of dataset shapes. Your job is to convert raw data into one
of them; that's what a **formatter** does (`modeltune/formatters.py`).

```jsonc
// Language modeling: continued pre-training. Loss on everything.
{"text": "The mitochondria is the powerhouse of the cell..."}

// Prompt/completion (conversational): SFT, single turn. Loss on the completion only.      ← Project 1
{"prompt":     [{"role": "system", "content": "Extract entities..."}, {"role": "user", "content": "EU rejects German call..."}],
 "completion": [{"role": "assistant", "content": "{\"person\": [], \"organization\": [\"EU\"], ...}"}]}

// Messages: SFT, multi-turn. With assistant_only_loss, loss on every assistant turn.       ← Project 2, stage 1
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."},
              {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

// Preference: DPO / ORPO / ... (here the prompt is implicit: the shared prefix of chosen & rejected) ← Project 2, stage 2
{"chosen":   [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good answer"}],
 "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "worse answer"}]}

// Prompt-only + extra columns: GRPO. The reward functions read the extra columns.          ← Bonus A
{"prompt": [{"role": "user", "content": "Natalia sold clips to 48 friends..."}], "answer": "72"}
```

**Bring your own data in three steps:**
1. Put it in a `.jsonl` file (one JSON object per line) or on the Hub.
2. Write a formatter that maps one row to one of the shapes above (copy `alpaca` or `conll_to_ner_json`). Return
   `None` to drop a row.
3. Point the config at it: `data.dataset: mydata.jsonl`, `data.formatter: mymodule:myformatter`,
   `data.eval_split: 0.05` (carves 5% out for validation).

## 2. Chat templates: conversations → one token string

A model doesn't see "messages"; it sees one flat token sequence. The **chat template** (a Jinja program stored
with the tokenizer) defines the flattening. The same conversation in three families:

```text
Qwen3 / SmolLM (ChatML):
<|im_start|>system\nYou extract entities.<|im_end|>\n<|im_start|>user\nEU rejects German call<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{"person": []...}<|im_end|>\n

Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou extract entities.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nEU rejects German call<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{"person": []...}<|eot_id|>

Granite 3:
<|start_of_role|>system<|end_of_role|>You extract entities.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>EU rejects German call<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{"person": []...}<|end_of_text|>
```

Rules:
1. **Never hand-format.** Always call `tokenizer.apply_chat_template(messages, ...)`.
2. **Train and serve with the same template.** A model fine-tuned on ChatML and served with another format sees
   text it never saw, and behaves erratically.
3. `add_generation_prompt=True` appends the assistant header (`<|im_start|>assistant\n`) so the model knows it
   should speak next. Use it at inference, never when rendering a full training example.
4. Template kwargs change the string: Qwen3's `enable_thinking=False` appends an empty `<think>\n\n</think>\n\n`
   block. The prompt/completion TRL builds for Project 1 therefore makes the model learn *"empty think block, then
   JSON"*, which matches inference with `enable_thinking: false` (set in the config's `eval.chat_template_kwargs`).
   Look for it in the tokenized example `modeltune.train` prints.

### Base models have no template (Project 2)
SmolLM3-3B-**Base** ships with `chat_template = None`, no pad token, and EOS `<|end_of_text|>` (meaning "end of
document"). We supply `templates/chatml.jinja` and set `eos_token: <|im_end|>`. Two details matter:
- We **reuse tokens already in the vocabulary**. `<|im_start|>`/`<|im_end|>` exist in SmolLM3's tokenizer with
  trained-looking embeddings (norm ≈ 7.7 vs a vocab mean of 7.0; we checked). New tokens would start untrained,
  and LoRA can't train embedding rows unless you add `trainable_token_indices`/`modules_to_save` (chapter 05).
  `modeltune` refuses an `eos_token` that isn't in the vocab for this reason.
- The assistant part is wrapped in `{% generation %}…{% endgeneration %}`, which lets the tokenizer return an
  *assistant mask*, so TRL can compute the loss only on assistant tokens (`assistant_only_loss: true`).

## 3. Special tokens and the five classic bugs

| Token | Role |
|---|---|
| BOS | start of sequence (some models only) |
| **EOS / end-of-turn** | "I'm done". Generation stops when it's produced |
| PAD | filler to make a batch rectangular; must be excluded from the loss |
| role markers | `<|im_start|>`, `<|start_header_id|>`, … |

1. **No EOS in the training target → the model never stops.** The template must close the assistant turn with
   the token generation will stop on, and that token must be in the loss. (Check the printed training example:
   the loss block should end with `<|im_end|>`.)
2. **pad = eos and padding masked by token id → EOS is masked too → never stops.** TRL masks padding using the
   attention mask, not the token id, so `pad_token = eos_token` is safe *here*. Hand-written collators often
   get this wrong.
3. **Double BOS:** the template already inserted BOS, then `tokenizer(text)` adds another. When tokenizing
   templated text, pass `add_special_tokens=False` (`modeltune/inference.py` does).
4. **Template mismatch** between training, evaluation and serving (including thinking on/off).
5. **Wrong `eos_token_id` in `generation_config`:** the tokenizer knows about `<|im_end|>` but `model.generate`
   stops on the old id. `modeltune.modeling.load_model` syncs both.

## 4. Loss masking: train on the answer, not the question

SFT minimizes next-token loss, but *which* tokens? Labels set to `-100` are ignored by PyTorch's cross-entropy.

| Setting | Loss on | When |
|---|---|---|
| `completion_only_loss` (default for prompt/completion data) | completion tokens | single-turn tasks (Project 1) |
| `assistant_only_loss: true` | every assistant turn | multi-turn chat (Project 2); needs `{% generation %}` tags |
| neither | all tokens | continued pre-training; or deliberately learning the prompts too |

Why mask? In Project 1, the system prompt plus the sentence is ~100 tokens and the JSON is ~35. Without masking,
~75% of the gradient would go into memorizing a fixed system prompt and predicting news sentences, which is not the
skill we want. In chat data, training on user turns teaches the model to *write user turns*: you'd see it
continue a conversation by playing both sides. (Nuance: Shi et al., 2024 found loss on instructions can help when
completions are very short and data is tiny. It's a hyperparameter, not a law.)

`modeltune.train` prints one tokenized example and **the exact tokens that receive loss**. For Project 1 on
Qwen3 it's about 34 of 134 tokens: `<think>\n\n</think>\n\n{...json...}<|im_end|>\n`. **Always read this
printout.** It catches half the bugs in this chapter.

## 5. Length, truncation, padding, packing

- Look at the **token-length distribution** of your formatted data before choosing `max_length`. Truncating the
  end of an answer teaches the model to stop mid-sentence (and drops its EOS).
- Batches are padded to the longest example in the batch; very uneven lengths waste compute on padding.
- **Packing** concatenates examples into full `max_length` rows. It's much faster for short examples, but
  examples can attend across boundaries unless the attention kernel supports per-example masking
  (FlashAttention with `padding_free`, Ampere+). On a T4, prefer plain padding unless speed is critical.

## 6. Quality, quantity, coverage

**Quality dominates.** LIMA: 1,000 curated examples beat much larger noisy sets. Concretely:
- **Consistency:** the same kind of input always gets the same kind of output (same JSON key order, same tone).
  Inconsistent targets get averaged into mush.
- **Correctness:** the model will learn your label errors. Sample 50 rows and *read them*.
- **Coverage:** include edge cases you care about. CoNLL has many sentences with no entities, which teaches the
  model to output empty lists instead of inventing entities.
- **Dedup and decontamination:** remove near-duplicates, and make sure no test example (or paraphrase) is in training.

**How much?** Rough starting points: a narrow format/task needs 500–5k examples; a new general behavior (chat)
10k–100k+; preference pairs 5k–60k; RL prompts 1k–10k. Run a **data-scaling sweep** (e.g. 500 / 2k / 8k) to see
whether more data still helps. Project 1's notebook includes one.

## 7. Splits and leakage

- **train**: what the model learns from. **validation**: what you watch during training and tune
  hyperparameters on. **test**: touched once, at the end, for the number you report.
- Every decision made by looking at the test set leaks it. Project 1 uses CoNLL's `validation` during training and
  `test` in `modeltune.evaluate`.
- Split by *source* when examples are correlated (all sentences from one article go to the same split).

## 8. Synthetic data and distillation

Most modern instruction data is generated by stronger models (Capybara, UltraFeedback and SmolTalk all are). Recipes:
- **Distill:** write prompts (or collect real ones), have a strong teacher answer, filter, then SFT.
- **Self-Instruct / Magpie:** have a model invent instructions, then answers.
- **Rejection sampling:** sample N answers, keep the ones a checker/judge accepts (the bridge to RL, chapter 12).

Risks: **diversity collapse** (everything sounds the same), inherited errors, the teacher's licence or terms of
use, and benchmark contamination. Filter hard, dedupe, and evaluate on *real* data.

## 9. Mixing to avoid forgetting

If the fine-tuned model must remain a generalist, mix in 5–20% general instruction data (e.g. a slice of a broad
SFT set). With `modeltune` you can build a mixed JSONL offline and point `data.dataset` at it.

## Check yourself
1. Your fine-tuned chat model keeps writing "User: …" lines after its answer. Name two likely causes.
2. Why does Project 1 use prompt/completion while Project 2 uses messages + `assistant_only_loss`?
3. You add a new special token `<answer>` and fine-tune with LoRA on `all-linear`. What's wrong with its embedding, and two ways to fix it?
4. Your examples are 200–3,000 tokens and you set `max_length: 512`. What exactly does the model learn from the long examples?
5. Why is `add_special_tokens=False` important when tokenizing text that came out of `apply_chat_template`?

**Next:** [07 · Training dynamics](07_training_dynamics.md)
