# 00 · Start here

> **Goal of the course:** after it, you can look at any open model and any dataset and know *whether* to
> fine-tune, *which method* to use, *how much GPU* you need, *how to set the knobs*, *how to tell if it worked*,
> and *how to fix it when it didn't*, and you'll have the code to do it.

## How this course is built

Fine-tuning is **concept-heavy, code-light**. The code you run is ~1,000 lines and the same for every project.
What changes between a good and a bad fine-tune is understanding: data formatting, loss masking, memory,
learning rates, evaluation. So the course is mostly chapters you read, with projects that make each idea concrete.

```
Part A · Foundations         00 start here → 01 LLM refresher → 02 pre-training (build TinyGPT) → 03 post-training map
Part B · The toolkit         04 GPU memory & precision → 05 LoRA/QLoRA → 06 data & chat templates
                             → 07 training dynamics → 08 evaluation
Part C · Project 1           09 Qwen3-0.6B → strict JSON extractor (LoRA + SFT)
Part D · Alignment           10 preference optimization theory → 11 Project 2: SmolLM3-3B-Base → SFT → DPO (QLoRA)
Part E · Frontier topics     12 RL with verifiable rewards (GRPO) → 13 Mixture of Experts (+ fine-tune one)
Part F · Real world          14 shipping (merge, quantize, serve) → 15 playbook & cheatsheet
```

**The two projects are designed to differ on every axis**, so you learn what transfers and what doesn't:

| Axis | Project 1 | Project 2 |
|---|---|---|
| Starting point | an **instruct** model (already chats) | a **base** model (only continues text) |
| Goal | one narrow skill (text → strict JSON) | a general behavior (be a helpful, preferred assistant) |
| Method | SFT | SFT **then** DPO |
| Data shape | prompt → completion, single turn | multi-turn conversations, then preference pairs |
| Loss masking | completion-only | assistant-turns-only |
| Memory strategy | LoRA on 16-bit weights | **Q**LoRA on 4-bit weights |
| Model family | Qwen3 (ChatML with `<think>`) | SmolLM3 (no template, we supply one) |
| Evaluation | exact, programmatic (F1, JSON validity) | fuzzy (reward-model score, length, reading outputs) |

The bonus projects add the remaining two big ideas: **reinforcement learning** (GRPO) and **Mixture-of-Experts**.

## The loop you will repeat in every project

```
  1. Look at the data            (print it AFTER formatting, exactly as the model sees it)
  2. Measure the baseline        (python -m modeltune.evaluate CONFIG --base)
  3. Train                       (python -m modeltune.train CONFIG)
  4. Measure again               (python -m modeltune.evaluate CONFIG)
  5. Read the failures           (not the averages; the actual wrong outputs)
  6. Change ONE thing, go to 3   (and write the result down)
```

People who are good at this mostly differ in steps 1, 2 and 5. Most failed fine-tunes are data or evaluation
bugs, not modeling problems.

## Should you fine-tune at all?

Fine-tuning is not the first tool. Climb this ladder and stop at the first rung that works:

| Rung | What it is | Use it when | Cost |
|---|---|---|---|
| 1. Prompting | instructions + a few examples in the prompt | the model *can* do it and just needs to be told | minutes |
| 2. RAG | retrieve documents and put them in the prompt | the model needs **facts** it doesn't have, or facts that change | days |
| 3. Fine-tuning | update the weights on examples | you need a **behavior, format, style or skill** applied reliably, cheaply, at scale | days–weeks |
| 4. Continued pre-training | more next-token training on raw domain text | a whole new domain or language (lots of unlabeled text) | weeks, more compute |
| 5. Pre-training | from scratch | you are a lab | $$$$ |

The key sentence: **fine-tuning teaches *how*, RAG supplies *what*.** Fine-tuning is good at format
("always valid JSON"), style and tone, task specialization (a 0.6B model doing one job as well as a 70B
model prompted), latency/cost (small fine-tuned model instead of a big prompted one), and behaviors that are hard
to describe but easy to demonstrate. It is unreliable at injecting new facts: models learn new knowledge
slowly through fine-tuning, and trying to do so tends to increase hallucination (Gekhman et al., 2024,
*"Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"*). Use RAG for facts.

Signs that fine-tuning is the right call:
- Your prompt has grown to pages of rules and the model still breaks them sometimes.
- You need a small/cheap/local model to do what a big API model does.
- You have (or can make) hundreds to thousands of examples of the desired behavior.
- Output must be machine-parsed (JSON, SQL, function calls) with near-zero failure rate.

## Setup

**Colab (recommended to start):** open any notebook in `notebooks/` from GitHub with the "Open in Colab" badge,
set *Runtime → Change runtime type → T4 GPU*, add your Hugging Face token (with **write** access) in the
🔑 *Secrets* panel as `HF_TOKEN`, and run top to bottom.

**Your own GPU:** install [uv](https://docs.astral.sh/uv/), then `uv sync` in the repo (it builds `.venv` from
`uv.lock`, so you get exactly the tested versions), and run `uv run python tests/smoke_test.py` to check that
everything is wired up. Throughout the course, commands are written as `python -m modeltune...`; locally,
prefix them with `uv run` (or activate `.venv` once). Colab notebooks install with
`uv pip install --system -r pyproject.toml`, which keeps Colab's preinstalled CUDA PyTorch.

**Pushing to the Hub:** configs push to `57Ajay/...`. If you fork this, change `hub.repo_id` (or override it:
`hub.repo_id=yourname/whatever`), or set `hub.push=false`.

## How to read each chapter

Each chapter has: **why** (the problem it solves), **what** (the concept, with the math only where it buys
intuition), **how** (where it shows up in our code and configs), **pitfalls**, and **check yourself**
questions. Don't skip the questions. If you can't answer one, re-read; the projects assume you can.

A good habit: keep a `LOG.md` with one line per experiment: `date · config · change · metric · minutes · notes`.
After the course it becomes your personal fine-tuning playbook.

## Check yourself
1. A support bot must answer questions about a product catalog that changes weekly. Fine-tune, RAG, or both? Why?
2. Why might a fine-tuned 0.6B model beat a prompted 8B model on one narrow task, but lose badly on everything else?
3. What is the difference between Project 1's and Project 2's starting model, and why does it change the method?

**Next:** [01 · LLM refresher](01_llm_refresher.md)
