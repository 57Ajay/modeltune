# 08 · Evaluation: did it actually get better?

> **Why this chapter:** training loss tells you the model imitates the training data. It doesn't tell you the
> model is *useful*, that it didn't break something else, or that the improvement is bigger than noise.
> Evaluation is where you find out whether to ship, iterate, or stop.

## 1. Four levels of evaluation

| Level | Question | Tooling in this course |
|---|---|---|
| 1. **Task metric** | does it do *the job* on held-out data? | `modeltune.evaluate` + `metrics.py` |
| 2. **Regression** | did general abilities survive? | a few benchmarks via lm-evaluation-harness |
| 3. **Qualitative** | do the outputs look right to a human? | reading `eval_*.json` samples, `modeltune.chat` |
| 4. **Production** | latency, cost, failure rate, user outcomes | logging after deployment (chapter 14) |

Do level 1 for every run, level 3 always (it takes 10 minutes and catches what metrics miss), level 2 before you
ship, and level 4 once real users exist.

## 2. Always start with a baseline

`python -m modeltune.evaluate CONFIG --base` runs the *unchanged* model on the same prompts with the same
decoding. Without it, "F1 = 0.85" means nothing. For a fair comparison:
- **Same prompts, same template, same decoding** (greedy, same `max_new_tokens`, same `chat_template_kwargs`).
  `evaluate` uses one code path for base and fine-tuned.
- Consider a **strong baseline** too: a well-engineered few-shot prompt, or a bigger model via API. If prompting
  gets 95% of the way, fine-tuning may not be worth maintaining.

## 3. Designing a task metric (Project 1 as the example)

The deliverable is "valid JSON with the right entities", so we measure each part separately:

| Metric | Definition | Why it exists |
|---|---|---|
| `json_strict` | the *entire* output parses as JSON | what a downstream `json.loads` needs. Prompted models often wrap JSON in prose or code fences |
| `json_parseable` | JSON found after stripping fences/`<think>` | separates "knows the content" from "knows the format" |
| `exact_match` | predicted entity set == gold set | strict all-or-nothing per sentence |
| `precision` / `recall` / **`f1`** | over (type, entity-string) pairs, micro-averaged | partial credit; precision ↓ = hallucinated entities, recall ↓ = missed entities |

General principles: measure what the user of the output cares about; decompose it so a number tells you *what*
to fix; make it **automatic and deterministic** so you can run it after every change.

For **open-ended** outputs (Project 2) there's no single right answer, so we use a **reward model** score plus
answer length and read the outputs (§6). For **checkable** outputs (Bonus A: math) we parse the final number and
compare (`metrics.gsm8k`).

## 4. Is the difference real? (noise)

A metric over n examples has sampling error. For a rate *p*, the standard error is `√(p(1−p)/n)`:

| n | p = 0.80 | 95% interval ≈ |
|---|---|---|
| 100 | ±4.0% SE | ±8% |
| 300 | ±2.3% SE | ±4.5% |
| 1,000 | ±1.3% SE | ±2.5% |

So with 300 test examples, 0.82 vs 0.80 is noise; 0.90 vs 0.80 is real. Because base and fine-tuned models are
scored on the **same examples**, you can also look at *paired* differences (examples that flipped), which is more
sensitive. Use enough examples (`eval.max_samples`) for the precision you need, and report n.

## 5. Benchmarks for regression testing

The fine-tuned model should still be a decent general model, unless you only ever run it for one task.

| Benchmark | Tests | Kind |
|---|---|---|
| MMLU / MMLU-Pro | knowledge across 57+ subjects | multiple choice |
| GSM8K / MATH | math word problems | exact answer |
| HumanEval / MBPP | code | unit tests |
| **IFEval** | instruction following ("answer in 3 bullet points, no commas") | programmatic checks |
| MT-Bench, AlpacaEval, Arena-Hard | open-ended chat quality | LLM judge |

Run a small slice with EleutherAI's **lm-evaluation-harness** (it can load a PEFT adapter directly):

```bash
# `--with` adds lm-eval to a temporary environment on top of the project's, without changing pyproject.toml
uv run --with lm-eval lm_eval --model hf \
        --model_args pretrained=Qwen/Qwen3-0.6B,peft=57Ajay/qwen3-0.6b-ner-json-lora \
        --tasks ifeval,gsm8k --limit 200 --batch_size 16
```
Run once with and once without `peft=` and compare. Caveats: public benchmarks may be **contaminated** (seen in
pre-training), scores depend heavily on prompt format and few-shot settings, and small models have high variance
on them.

## 6. Judges: LLMs and reward models

**LLM-as-a-judge:** ask a strong model to grade (pointwise, 1–10 against a rubric) or compare (pairwise A vs B).
It scales and correlates reasonably with humans, but has known biases:
- **position bias:** prefers the first (or second) answer, so evaluate both orders;
- **verbosity bias:** prefers longer answers, so control for or report length (AlpacaEval's "length-controlled" win rate exists for this);
- **self-preference:** prefers outputs that look like its own.

**Reward models** (what Project 2 uses): a model trained on human preference pairs to output a scalar "how much
would a human like this answer". We use `Skywork/Skywork-Reward-V2-Qwen3-0.6B`, small enough to run on a T4 right
after generation. A quick sanity check we ran: *"The capital of France is Paris."* scored **+2.6**, while
*"I like turtles. Bananas are yellow."* scored **−8.0** for the same question. Limits:
- It's a *proxy*. Optimizing against it (even indirectly, via DPO on similar data) can exploit its quirks
  (**reward hacking**). The classic quirk is length, so `evaluate` always reports `avg_answer_words`.
- Compare scores **only across models on the same prompts with the same reward model**. Absolute values mean nothing.

## 7. Error analysis: the highest-value activity

Averages hide *why*. After every eval:
1. Open `outputs/<run>/eval_finetuned.json` and read 20–50 failures (notebook 01 has a cell for this).
2. **Bucket them:** for NER, e.g. wrong type (ORG vs LOC: "Real Madrid"?), boundary errors ("Bank of England" vs
   "England"), hallucinated entities, missed entities, invalid JSON.
3. Count the buckets. The biggest one tells you what to fix: label noise → clean data; boundary errors → more
   examples of that pattern; invalid JSON → check truncation and EOS.
4. Some "errors" are **gold-label errors**. Every real dataset has them, and they cap your achievable score.

## 8. How `modeltune.evaluate` works

```
load_eval_split  →  generate (greedy, left-padded batches)  →  free GPU  →  METRICS[eval.metric]  →  eval_<tag>.json
```
- `--base` = no adapter; default = `outputs/<run_name>` if it exists, else the Hub repo; `--adapter X` = anything else.
- References come from the `completion` (SFT data) or `answer` column (GRPO data).
- The JSON file stores the results **and every prompt/prediction/reference**, so you can always go back to the raw outputs.
- Add your own metric: write `def my_metric(prompts, predictions, references, cfg, hw) -> dict`, register it in
  `METRICS`, set `eval.metric: my_metric`.

## Check yourself
1. Your fine-tuned model has `json_parseable = 0.99` but `json_strict = 0.60`. What is it doing, and is that a training or an inference problem?
2. Base F1 0.71, fine-tuned F1 0.74 on 100 examples. Ship it? What would you do first?
3. DPO raised the reward-model score a lot, and answers got 2× longer. What do you check next?
4. Why must base and fine-tuned evaluation use identical decoding settings and templates?

**Next:** [09 · Project 1: JSON extractor](09_project1_json_extractor.md)
