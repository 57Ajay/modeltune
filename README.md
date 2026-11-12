# modeltune: learn LLM fine-tuning from 0 to 100

A hands-on course plus a small, readable toolkit. You start by **pre-training a tiny GPT from scratch**,
then work through the theory (memory math, LoRA/QLoRA, data and chat templates, training dynamics, evaluation),
and then **fine-tune real modern models** with different methods, models and data:

| | Project | Model | Method | Data | Notebook |
|---|---|---|---|---|---|
| 🧱 | Pre-training warm-up | TinyGPT (~12M, from scratch) | next-token prediction | TinyStories | [00](notebooks/00_pretrain_tiny_gpt.ipynb) |
| 1️⃣ | Strict-JSON entity extractor | Qwen3-0.6B (instruct) | **LoRA + SFT** (prompt→completion) | CoNLL-2003 → JSON | [01](notebooks/01_project1_ner_json_lora.ipynb) |
| 2️⃣ | Base model → aligned assistant | SmolLM3-3B-**Base** | **QLoRA + SFT → DPO** | Capybara, UltraFeedback | [02](notebooks/02_project2_sft_dpo_qlora.ipynb) |
| ➕ | Reasoning with RL | Qwen3-0.6B | **LoRA + GRPO** (verifiable rewards) | GSM8K | [03](notebooks/03_bonus_grpo_gsm8k.ipynb) |
| ➕ | Mixture-of-Experts | Granite-3.1-3B-A800M (MoE) | **LoRA on experts + router aux loss** | same as Project 1 | [04](notebooks/04_bonus_moe_granite.ipynb) |

Everything runs on a **free Colab T4 (16 GB)** and on any newer GPU without code changes: the code detects
the GPU and picks fp16 or bf16 itself.

## The idea: one YAML file = one fine-tune

```bash
python -m modeltune.evaluate configs/p1_qwen3_ner_json_lora.yaml --base   # 1. measure the baseline
python -m modeltune.train    configs/p1_qwen3_ner_json_lora.yaml          # 2. fine-tune
python -m modeltune.evaluate configs/p1_qwen3_ner_json_lora.yaml          # 3. measure again
python -m modeltune.chat     configs/p1_qwen3_ner_json_lora.yaml --prompt "Sundar Pichai visited Paris."
python -m modeltune.merge    configs/p1_qwen3_ner_json_lora.yaml --push   # 4. ship a standalone model
```

To fine-tune **a different model**, change `model.id`. To use **different data**, change the `data` block
(write a 10-line formatter if your columns are unusual). To switch **method**, set `method: sft | dpo | grpo`.
Any field can also be overridden from the command line:

```bash
python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml model.id=Qwen/Qwen3-1.7B lora.r=32 run_name=p1-1.7b
```

## The course

Read in order; every chapter explains **why** before **how**, and ends with self-check questions.

| # | Chapter | What you'll understand |
|---|---|---|
| 00 | [Start here](course/00_start_here.md) | the map; prompt vs RAG vs fine-tune; how to use this repo |
| 01 | [LLM refresher](course/01_llm_refresher.md) | tokens → attention → logits; RoPE, RMSNorm, SwiGLU, GQA, KV cache, decoding |
| 02 | [Pre-training](course/02_pretraining.md) | the objective, data, tokenizers, scaling laws, compute; build TinyGPT |
| 03 | [The post-training landscape](course/03_post_training_landscape.md) | CPT, SFT, RLHF, DPO, RLVR, distillation, and when to use which |
| 04 | [GPU memory & precision](course/04_gpu_memory_and_precision.md) | where every byte goes; fp16 vs bf16 (the T4 story); checkpointing; multi-GPU |
| 05 | [PEFT: LoRA & QLoRA](course/05_peft_lora_qlora.md) | the math, the knobs (r, α, targets), NF4, variants, LoRA vs full FT |
| 06 | [Data & chat templates](course/06_data_and_chat_templates.md) | formats, templates, special tokens, loss masking, quality, synthetic data |
| 07 | [Training dynamics](course/07_training_dynamics.md) | hyperparameters, reading loss curves, forgetting, a debugging checklist |
| 08 | [Evaluation](course/08_evaluation.md) | task metrics, baselines, benchmarks, LLM judges, reward models, noise |
| 09 | [Project 1: JSON extractor](course/09_project1_json_extractor.md) | LoRA + SFT end to end on Qwen3-0.6B |
| 10 | [Preference optimization](course/10_preference_optimization.md) | RLHF/PPO, DPO derivation, IPO/KTO/ORPO/SimPO, failure modes |
| 11 | [Project 2: base → aligned](course/11_project2_base_to_aligned.md) | QLoRA SFT then DPO on SmolLM3-3B-Base |
| 12 | [RL & reasoning (GRPO)](course/12_rl_grpo_reasoning.md) | policy gradients, GRPO, verifiable rewards, reward hacking |
| 13 | [Mixture of Experts](course/13_mixture_of_experts.md) | routers, experts, load balancing, and fine-tuning MoEs |
| 14 | [Shipping](course/14_shipping.md) | merging, quantizing, serving (vLLM, GGUF), model merging, safety |
| 15 | [Playbook & cheatsheet](course/15_playbook_cheatsheet.md) | fine-tune *any* model: recipe, defaults, troubleshooting table |

## Repository layout

```
course/        the chapters (theory-heavy, read these)
notebooks/     Colab notebooks: thin wrappers around the scripts below
configs/       one YAML per run: model, LoRA, data, training args, eval, hub
modeltune/     the toolkit (~1,000 lines, much of it comments): config, hardware, data, formatters, modeling,
               train (SFT/DPO/GRPO), inference, evaluate, metrics, rewards, chat, merge
pretrain/      tiny_gpt.py: pre-training from scratch in one file
templates/     chatml.jinja: a chat template for base models (with assistant-loss tags)
tests/         smoke_test.py: runs every config end-to-end on CPU with tiny models
pyproject.toml dependencies (pinned core libraries)  ·  uv.lock: the full locked environment
```

## Running locally / on other GPUs

The project is managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml` declares the dependencies and
`uv.lock` pins every package, so every machine gets exactly the same environment.

```bash
uv sync                                   # create .venv from uv.lock (Linux gets a CUDA build of PyTorch)
uv run python tests/smoke_test.py         # ~5 min on CPU, checks that everything is wired up
uv run python -m modeltune.train configs/p1_qwen3_ner_json_lora.yaml
```

Every `python ...` command in this README and the course becomes `uv run python ...` locally (or activate
`.venv` once with `source .venv/bin/activate`). For Jupyter: `uv sync --group notebooks && uv run jupyter lab`.
In Colab, the notebooks install with `uv pip install --system -r pyproject.toml`, which keeps Colab's own CUDA
PyTorch.

* **T4 / V100** (compute capability < 8): fp16 + loss scaling, SDPA attention.
* **A100 / L4 / RTX 30xx-40xx / H100**: bf16 automatically. You can set `model.attn_implementation=flash_attention_2` if installed, and raise batch sizes.
* **Multi-GPU**: `uv run accelerate launch -m modeltune.train <config>` (data parallel; see chapter 04).
* **Windows**: PyPI ships CPU-only PyTorch for Windows; point uv at the PyTorch CUDA index (see uv's PyTorch guide).

The core libraries are pinned because the fine-tuning stack changes fast. For example, transformers v5 renamed
`torch_dtype` → `dtype`, removed `warmup_ratio`, and stores MoE experts as fused tensors. Upgrade deliberately
(`uv lock --upgrade-package trl`), then run the smoke test.

## Results

Fill this in from your own runs (the `eval_*.json` files in `outputs/`). Writing down numbers, including
failures, is half of the skill.

| Run | Metric | Before | After | Train time | Peak VRAM |
|---|---|---|---|---|---|
| P1 Qwen3-0.6B LoRA | entity F1 / strict JSON | | | | |
| P2 SmolLM3-3B SFT | reward-model score | | | | |
| P2 SmolLM3-3B SFT+DPO | reward-model score | | | | |
| GRPO Qwen3-0.6B | GSM8K accuracy | | | | |
| MoE Granite 3B-A800M | entity F1 / strict JSON | | | | |

## Licenses

Code: MIT. Models and datasets keep their own licenses: Qwen3 (Apache-2.0), SmolLM3 (Apache-2.0),
Granite 3.1 (Apache-2.0), CoNLL-2003 (Reuters RCV1 terms, research use, so this repo builds the JSON version
on the fly instead of redistributing it), Capybara (see its dataset card), UltraFeedback (MIT), GSM8K (MIT),
TinyStories (CDLA-Sharing-1.0).
Check a license before you publish a fine-tune.
