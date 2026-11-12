# 09 · Project 1: Qwen3-0.6B → a strict-JSON entity extractor (LoRA + SFT)

> **What you'll build:** a 0.6B model that reads a sentence and returns *only* a JSON object of the people,
> organizations, locations and other names in it, reliably enough to feed straight into `json.loads`.
> **What you'll learn:** the complete fine-tuning loop on the simplest serious setup: an instruct model, single-turn
> prompt/completion data, LoRA on a 16-bit base, and an exact programmatic metric.
>
> Notebook: [`notebooks/01_project1_ner_json_lora.ipynb`](../notebooks/01_project1_ner_json_lora.ipynb) ·
> Config: [`configs/p1_qwen3_ner_json_lora.yaml`](../configs/p1_qwen3_ner_json_lora.yaml) ·
> Runtime: ~10–15 min training on a T4 (estimate)

## 1. Why this task

Structured extraction ("turn messy text into a fixed schema") is one of the most common reasons companies
fine-tune: invoices → fields, emails → tickets, resumes → profiles, logs → events. It's also ideal for learning:
- **Prompting gets you partway, not all the way.** Small models know what a person or city is, but they break
  the format (prose around the JSON, code fences, extra keys) and are inconsistent about edge cases.
- **The metric is exact.** No judges, no vibes. You know immediately whether a change helped.
- **It's fast.** Inputs are one sentence and outputs ~35 tokens, so you can run many experiments.

Named-entity recognition (NER) is classically done with encoder models (BERT-style token classifiers, ~93 F1 on
CoNLL-2003 with exact-span scoring). A generative LLM is more flexible: change the schema in the prompt and data,
and the same recipe extracts anything. Our metric (sets of entity strings per sentence) isn't directly comparable
to the token-level literature, so compare only against your own baseline.

## 2. Design decisions (and why)

| Decision | Choice | Reasoning |
|---|---|---|
| Model | **Qwen3-0.6B** (instruct, hybrid thinking) | modern, Apache-2.0, tiny enough for many fast experiments; already understands instructions, so we're teaching a *format and a task*, not language |
| Method | **SFT** | we have exact target outputs; imitation is the right tool |
| PEFT | **LoRA r=16, all-linear, on a 16-bit base** | 0.6B × 2 B = 1.2 GB fits easily, so QLoRA would only add quantization error and slowness |
| Data format | **prompt/completion** (conversational) | single turn; loss only on the JSON (`completion_only_loss` defaults on) |
| Thinking | **off** (`enable_thinking: false`) | we want the answer immediately; also keeps train and inference templates identical |
| Decoding | greedy | extraction has one right answer; sampling only adds errors |

## 3. The data: CoNLL-2003 → JSON

CoNLL-2003 (Reuters news, 1996) labels each token with a **BIO** tag: `B-X` begins an entity of type X, `I-X`
continues it, `O` is outside any entity. Types: PER, ORG, LOC, MISC (nationalities, events, other names).

```
tokens : EU    rejects German call to boycott British lamb .
tags   : B-ORG O       B-MISC O    O  O       B-MISC  O    O
target : {"person": [], "organization": ["EU"], "location": [], "misc": ["German", "British"]}
```

`modeltune/formatters.py:conll_to_ner_json` does three things:
1. **Drops junk rows** (returns `None`): `-DOCSTART-` separators and fragments under 3 tokens. That leaves
   12,733 train / 2,970 validation / 3,143 test sentences.
2. **Decodes BIO spans** into entity strings and **detokenizes** ("U.S. , he said ." → "U.S., he said.").
3. Builds the **prompt** (system instructions + sentence) and the **completion** (compact JSON, fixed key order).

Notice the deliberate data choices: **fixed key order** and **all four keys always present** (consistency means
the model never has to guess the schema), entities **in order of appearance** (a deterministic target), and
**sentences with no entities kept** (the model must learn to output empty lists rather than invent entities).

We don't re-upload a processed copy: CoNLL-2003 is Reuters text under research terms, so the formatter rebuilds
it on the fly from `eriktks/conll2003` (its parquet branch, `refs/convert/parquet`, because script-based datasets
no longer load in `datasets` 4+).

## 4. The config, line by line

```yaml
model:
  id: Qwen/Qwen3-0.6B
  quantization: none                # 16-bit base; see §2
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}   # ≈10.1M trainable params (chapter 05)
data:
  dataset: eriktks/conll2003
  revision: refs/convert/parquet
  formatter: modeltune.formatters:conll_to_ner_json
  eval_split: validation            # watched during training
  max_train_samples: 4000           # 250 steps at batch 16, plenty for a narrow task
train:
  learning_rate: 2.0e-4             # LoRA LR (~10× full FT)
  lr_scheduler_type: cosine
  warmup_steps: 0.05
  per_device_train_batch_size: 16
  max_length: 512                   # longest example ≈ 250 tokens
  eval_steps: 50
eval:
  metric: ner_json
  split: test                       # touched only for the final number
  chat_template_kwargs: {enable_thinking: false}
```

Everything the course talks about is visible in this file. To fine-tune another model on the same task, change
`model.id`. To change the task, write a new formatter.

## 5. Run it, and what to look for

**Step 1: look at the data** (notebook cell 1). Read a few prompt/completion pairs. Do the targets look right?

**Step 2: baseline.** `python -m modeltune.evaluate configs/p1_qwen3_ner_json_lora.yaml --base`
What you'll typically see from a prompted 0.6B model: decent `json_parseable` but lower `json_strict` (extra text,
code fences), and moderate F1 with type confusions, missed entities, and sometimes invented keys or entities.
Write the numbers down.

**Step 3: train.** `python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml`. Read the output top to bottom:
- `trainable params: ~10.1M || all params: ~606M || trainable%: ~1.7`
- the **tokenized example**, where the loss block is only `<think>\n\n</think>\n\n{...}<|im_end|>\n`
  (~34 of ~134 tokens). If you see the system prompt in the loss block, masking is broken.
- `loss` should fall fast (JSON braces and key names are learned within a few dozen steps) and `eval_loss` should
  follow. `mean_token_accuracy` should climb high.
- at the end: training time and **peak GPU memory** (a few GB). Compare with your chapter 04 estimate.

**Step 4: evaluate again** (without `--base`). Expect `json_strict` near 1.0: the format problem is essentially
solved by SFT. F1 should rise clearly above the baseline. The remaining errors are the *content* errors that are
genuinely hard.

**Step 5: read the failures.** The notebook prints sentences where the prediction isn't an exact match. Bucket
them (chapter 08 §7). You'll find boundary disagreements ("Bank of England" vs "England"), type ambiguity (a
country name used for a sports team), MISC confusion (the fuzziest category), and some gold-label errors.

## 6. Using the model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from modeltune.formatters import NER_INSTRUCTIONS

repo = "57Ajay/qwen3-0.6b-ner-json-lora"          # or outputs/p1-qwen3-0.6b-ner-json-lora
model = AutoPeftModelForCausalLM.from_pretrained(repo, device_map="auto")
tok = AutoTokenizer.from_pretrained(repo)

messages = [{"role": "system", "content": NER_INSTRUCTIONS},
            {"role": "user", "content": "Sundar Pichai said Google will open an office in Gurugram."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`python -m modeltune.merge CONFIG --push` produces a standalone model (`…-merged`) that needs no PEFT. Note that the
**system prompt is part of the learned interface**: the model was trained with that exact instruction text, so
send the same one in production.

**Constrained decoding** (JSON-schema-guided generation in vLLM, Outlines, etc.) can *guarantee* syntactically valid
JSON even from a base model. It doesn't make the *content* correct. Fine-tuning and constrained decoding
complement each other.

## 7. Experiments (each teaches one idea)

Run with overrides and a new `run_name`, e.g.
`python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml lora.r=4 run_name=p1-r4 hub.push=false`,
then evaluate with the same overrides.

| Experiment | Override | Lesson |
|---|---|---|
| Rank | `lora.r=4` / `lora.r=64` (keep `alpha` = 2r) | narrow tasks need little capacity |
| Targets | `lora.target_modules=[q_proj,v_proj]` | attention-only LoRA is weaker (chapter 05) |
| LR | `train.learning_rate=2e-5` / `1e-3` | the most important knob; see how both fail |
| Data size | `data.max_train_samples=500` / `12000` | the data-scaling curve; when does more stop helping? |
| Epochs | `train.num_train_epochs=3` | watch eval loss vs F1 diverge (chapter 07) |
| Model size | `model.id=Qwen/Qwen3-1.7B` | does 3× the params beat more data? |
| Another family | `model.id=HuggingFaceTB/SmolLM2-360M-Instruct` | the config design: nothing else changes (check the printed template!) |
| Full fine-tune | `lora=null train.learning_rate=2e-5` | memory and quality vs LoRA (~10 GB on T4 with fp32 master weights) |
| Mask off | `train.completion_only_loss=false` | what happens when the loss includes the prompt? |

Keep a table: run · change · json_strict · F1 · minutes · peak GB.

## 8. What carries forward to Project 2

| Stays the same | Changes in Project 2 |
|---|---|
| the loop: data → baseline → train → eval → failures | **base** model instead of instruct: we supply a chat template and EOS |
| config-driven runs, overrides, `run_name` discipline | **QLoRA** because 3B in 16-bit is tight on a T4 |
| LoRA mechanics, LR ≈ 10× full FT | multi-turn `messages` + **assistant-only loss** |
| reading the tokenized example | fuzzy evaluation: reward model + reading outputs |
| `evaluate` / `chat` / `merge` commands | a **second stage** (DPO) that starts from the first stage's adapter |

## Check yourself
1. Why do we keep sentences with no entities in the training data?
2. Explain why `json_strict` usually jumps much more than F1 after fine-tuning.
3. You switch to `Qwen/Qwen3-0.6B-Base`. Its tokenizer *does* ship a ChatML template, but its `eos_token` is
   `<|endoftext|>`. What must you add to the config, and what would happen at inference without it? (chapter 06 §3)
4. Your colleague proposes QLoRA for this project "to save memory". Argue for or against.

**Next:** [10 · Preference optimization](10_preference_optimization.md)
