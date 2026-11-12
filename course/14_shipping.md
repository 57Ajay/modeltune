# 14 · Shipping: merge, quantize, serve, combine, and stay safe

> **Why this chapter:** a fine-tune that only runs in your notebook isn't finished. This chapter covers turning an
> adapter into something people can use: merged or not, quantized or not, on a GPU server or a laptop, plus the
> training-free trick of model merging and the safety and licensing questions you must answer before publishing.

## 1. Adapter or merged model?

| | Keep the LoRA adapter | Merge into the base |
|---|---|---|
| size | MBs | full model (GBs) |
| needs | PEFT at load time | nothing special |
| many tasks | one base + many hot-swappable adapters | one full copy per task |
| inference speed | tiny overhead (or zero with fused multi-LoRA kernels) | identical to the base |
| needed for | multi-LoRA serving | GGUF/llama.cpp, most quantization tools, simple deployment |

**Merging correctly** (`python -m modeltune.merge CONFIG [--push]`):
```
W_merged = W_base + (α / r) · B · A        (exact, done once per layer by merge_and_unload())
```
- Merge into the **original 16/32-bit base**, never into the 4-bit one. Re-quantizing after merging stacks a second
  rounding error on top of the first. `merge.py` reloads the base on CPU in its native dtype for this reason.
- A **QLoRA** adapter was trained against the *dequantized 4-bit* weights, so merging it into the exact 16-bit
  weights is slightly mismatched. It's usually harmless, but **re-run your evaluation on the merged model**.
- Merging is per adapter. Project 2's DPO adapter was trained *on top of* the SFT adapter (we continued training
  it), so its weights already include the SFT changes.

## 2. Model cards

TRL writes a basic `README.md` card when it pushes. Edit it before you share the model. A useful card states:
base model, training data (with licenses), method and key hyperparameters, **evaluation results with the
baseline**, intended use, known limitations and failure modes, the prompt/template to use (Project 1's system
prompt is part of its interface), and the license.

## 3. Serving options

| Tool | Where | Why |
|---|---|---|
| **transformers** `generate` / pipeline | anywhere | simplest; fine for evaluation and low traffic |
| **vLLM** | NVIDIA/AMD GPU servers | PagedAttention + continuous batching = high throughput; OpenAI-compatible API; **multi-LoRA** serving |
| **SGLang** | GPU servers | fast structured generation, prefix caching |
| **llama.cpp / Ollama / LM Studio** | CPU, laptops, Apple silicon | GGUF quantized models, runs anywhere |
| **MLX** | Apple silicon | native Metal performance |

```bash
# vLLM: serve the base and attach adapters by name (OpenAI-compatible endpoint on :8000)
vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules ner=57Ajay/qwen3-0.6b-ner-json-lora

# llama.cpp: merged model -> GGUF -> 4-bit, then run locally
python convert_hf_to_gguf.py outputs/p1-qwen3-0.6b-ner-json-lora/merged --outfile p1.gguf
./llama-quantize p1.gguf p1-Q4_K_M.gguf Q4_K_M
./llama-cli -m p1-Q4_K_M.gguf -p "..."
```
Whatever you serve with, use the **same chat template and generation settings** you evaluated with (chapter 06).
vLLM and llama.cpp read the template from the tokenizer files, which is one more reason `modeltune` saves the
tokenizer next to every adapter.

**Structured output in production:** for Project 1, combine the fine-tune with **constrained decoding** (vLLM's
guided/structured outputs with a JSON schema). Then validity is guaranteed by the decoder, and correctness comes
from the fine-tune.

## 4. Quantization for inference (post-training)

| Method | Bits | Notes |
|---|---|---|
| bitsandbytes (NF4/int8) | 4/8 | zero-effort, what QLoRA uses; slower kernels, mainly for training and experiments |
| **GPTQ** | 2–8 (usually 4) | layer-wise error-compensating quantization with calibration data |
| **AWQ** | 4 | protects the weights that matter most for activations; fast kernels |
| **GGUF k-quants** (Q8_0, Q6_K, Q4_K_M, …) | 2–8 | llama.cpp ecosystem; Q4_K_M is the popular default |
| **FP8** | 8 | near-lossless on Hopper+ GPUs; often the first thing to try on H100s |
| MXFP4 / NVFP4 | 4 | Blackwell-native 4-bit formats; gpt-oss ships in MXFP4 |

Rules of thumb: 8-bit is nearly lossless, 4-bit costs a little, and below 4 bits the loss becomes noticeable,
especially for small models (a 0.6B model has less redundancy to lose than a 70B one). **Measure your task metric
after quantizing**, since generic perplexity hides task-specific damage (e.g. JSON validity).

## 5. Model merging: combining fine-tunes without training

Fine-tunes of the *same base* live in a shared weight space, and you can often combine them arithmetically:

| Method | Idea |
|---|---|
| linear / "model soup" | average the weights of several fine-tunes |
| **task arithmetic** | task vector τ = θ_finetuned − θ_base; add (or subtract!) task vectors |
| **SLERP** | spherical interpolation between two models |
| **TIES** | trim small changes, resolve sign conflicts, then merge |
| **DARE** | randomly drop most of each delta and rescale the rest, which reduces interference |

Tool: **mergekit**. Real use: SmolLM3's team merged their preference-tuned checkpoint with a mid-training
checkpoint to recover long-context ability lost during alignment. Merging is cheap to try, though not guaranteed
to work, so evaluate the result like any other model.

## 6. Safety, privacy, licenses

- **Fine-tuning can remove safety behavior.** Qi et al. (2023) showed that fine-tuning an aligned model on a handful
  of harmful examples, and to a lesser degree even on *benign* data, degrades its refusals. If your model faces the
  public, evaluate refusal behavior before and after, and consider mixing safety data into training.
- **Models memorize.** Don't train on personal or confidential data you wouldn't want reproduced verbatim; dedupe,
  and remove secrets and PII.
- **Licenses travel.** Your fine-tune inherits the base model's license and must respect each dataset's terms
  (chapter 03 §7). Put both in the model card.

## 7. After launch: the data flywheel

1. **Log** inputs, outputs and outcomes (with consent): invalid-JSON rate, user corrections, thumbs up/down.
2. **Monitor drift:** production inputs change (new entity types, new languages); your test set should grow with them.
3. **Mine failures** into new training examples; thumbs up/down become KTO or DPO data (chapter 10).
4. **Version everything:** data snapshot + config (`modeltune_config.yaml`) + adapter repo revision, so any model
   can be rebuilt.

This loop, not any single training run, is how fine-tuned models get good.

## Check yourself
1. Why should you never merge an adapter into a 4-bit model and then ship that?
2. You need the Project 1 model on a laptop without a GPU. List the steps from adapter to running binary.
3. Two LoRA fine-tunes of the same base (a JSON extractor and a summarizer). Give two ways to serve both from one GPU.
4. Why might your entity F1 drop after Q4 quantization even though perplexity barely moved?

**Next:** [15 · Playbook & cheatsheet](15_playbook_cheatsheet.md)
