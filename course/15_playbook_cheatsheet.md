# 15 · Playbook & cheatsheet: fine-tune *any* model

> Keep this page open when you start a new fine-tune. It compresses the course into checklists, tables and a
> troubleshooting guide, with pointers back to the chapters.

## 1. The 12-step recipe (with this repo)

1. **Decide it's a fine-tuning problem** (ch. 00 ladder: prompt → RAG → fine-tune). Write the success metric first.
2. **Pick the method** (§2 below) and copy the closest config: P1 (SFT, single turn), P2 (SFT multi-turn / DPO),
   GRPO, or MoE.
3. **Pick the model:** set `model.id`. Run the *new-model checklist* (§3).
4. **Estimate memory** (ch. 04): weights + adapter states + activations. Choose LoRA vs QLoRA vs full.
5. **Format the data:** write a formatter (`modeltune/formatters.py` patterns), set `data.*`, hold out validation and test.
6. **Look at the data:** print 10 formatted rows (notebook 01 cell 1); check lengths; read 50 for label quality.
7. **Baseline:** `python -m modeltune.evaluate CONFIG --base`. Save the numbers.
8. **Dry run:** `python -m modeltune.train CONFIG train.max_steps=20 hub.push=false run_name=dry`. Read the tokenized
   example (loss only on the answer? ends with EOS?), trainable params, the first losses, and peak memory.
9. **Full run.** Watch loss, eval loss and grad norm (ch. 07).
10. **Evaluate + read failures** (ch. 08): metric vs baseline, bucket the errors.
11. **Iterate on one variable at a time:** data first, then LR, then capacity (r, model size), then epochs. Log every run.
12. **Ship** (ch. 14): merge (16-bit base), re-evaluate, quantize if needed and re-evaluate, write the model card.

## 2. Decision tables

**Method**

| You have | Goal | Method | Config to copy |
|---|---|---|---|
| (input, ideal output) pairs | task/format | SFT, prompt/completion | `p1_*` |
| conversations | chat behavior | SFT, messages + assistant_only_loss | `p2_*sft*` |
| (prompt, chosen, rejected) | preferred answers | DPO (after SFT) | `p2_*dpo*` |
| 👍/👎 on single answers | preferred answers | KTO | (TRL `KTOTrainer`) |
| prompts + a checker | accuracy on checkable tasks | GRPO | `bonus_grpo_*` |
| raw domain text | domain knowledge / language | continued pre-training (`{"text": ...}`, no masking) | `p1_*` with a text formatter |

**PEFT / precision** (16 GB GPU like a T4)

| Model size | Recommended | Notes |
|---|---|---|
| ≤ 1B | LoRA on 16-bit (full FT possible) | P1 |
| 1–4B | LoRA 16-bit if seq is short, else QLoRA | P2 uses QLoRA at 3B |
| 7–8B | QLoRA, batch 1–2 + accumulation, checkpointing | fits ~16 GB at seq 1–2k |
| ≥ 13B | QLoRA on a bigger GPU | or multi-GPU (FSDP) |
| MoE | memory = **total** params; experts may not quantize (ch. 13) | prefer small MoEs on T4 |

**Hardware**

| GPU | dtype (auto) | Extras |
|---|---|---|
| T4, V100 | fp16 + loss scaling | SDPA attention; check fp16 activation range for new models (ch. 04) |
| A10, L4, A100, RTX 30/40 | bf16 | can use `flash_attention_2` and packing with padding-free |
| H100+ | bf16 (FP8 possible) | larger batches; vLLM for GRPO generation |

## 3. New-model checklist

```python
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(MODEL); cfg = AutoConfig.from_pretrained(MODEL)
print(tok.chat_template is not None, tok.eos_token, tok.pad_token, tok.bos_token)
print(tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], tokenize=False))
print(cfg.architectures, getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None))
```
- [ ] **Chat template** present? If not, supply one built from existing tokens (`model.chat_template`).
- [ ] **EOS** is the *end-of-turn* token you want generation to stop on (`model.eos_token`)?
- [ ] **Thinking modes** (Qwen3 `enable_thinking`, `/think`): train and evaluate in the same mode.
- [ ] **Module names** for LoRA: `print(model)`; `all-linear` works for standard architectures.
- [ ] **MoE?** Experts fused? Then `target_parameters` for LoRA, keep the aux loss on, budget memory for total params.
- [ ] **fp16-safe** if you're on a T4 (Gemma-family models are known to overflow).
- [ ] **License** allows your use.

## 4. Hyperparameter defaults

| | LR | Epochs / steps | Eff. batch | Other |
|---|---|---|---|---|
| SFT LoRA | 1e-4 – 3e-4 | 1–3 | 16–64 | r 16–64, α = r…2r, all-linear, cosine, warmup 3–5% |
| SFT full | 1e-5 – 2e-5 | 1–3 | 64–256 | weight decay 0–0.1 |
| DPO LoRA | 5e-6 – 5e-5 | 1 | 16–64 | β 0.1; watch length & logps/chosen |
| DPO full | 3e-7 – 1e-6 | 1 | 32–128 | |
| GRPO LoRA | 1e-6 – 2e-5 | 100s of steps | 16–64 completions | G = 8–16, temperature ~1.0, β 0 – 0.04 |

## 5. Troubleshooting

| Symptom | Likely cause | Fix | Chapter |
|---|---|---|---|
| CUDA OOM | batch × seq too big; no checkpointing; 16-bit base too big | halve batch (×2 accumulation), `gradient_checkpointing`, lower `max_length`, `paged_adamw_8bit`, 4-bit | 04 |
| loss NaN / spikes | LR too high; fp16 overflow | lower LR; check `grad_norm`; bf16 hardware; filter bad rows | 04, 07 |
| loss flat from the start | everything masked; LR far too low; adapter not attached | read tokenized example; trainable params; LR sweep | 06, 07 |
| model never stops | EOS missing from targets / wrong EOS id / pad = eos masked by a custom collator | template closes the turn; `model.eos_token`; check the loss block ends with EOS | 06 |
| writes "User:" turns itself | loss on user turns; base model w/o SFT | `assistant_only_loss` / prompt-completion; do SFT | 06 |
| good in training, bad in `chat`/serving | template or kwargs mismatch (thinking mode!), different system prompt | identical template and system prompt everywhere | 06, 14 |
| JSON parseable but not strict | extra prose/fences from base habits | more SFT data/steps; constrained decoding at serve time | 08, 09 |
| great on task, broke general chat | forgetting | LoRA, lower LR, fewer epochs, mix general data, keep adapter separate | 03, 07 |
| eval loss ↑ but task metric ↑ | normal-ish for generative tasks | trust the task metric, but watch overfitting | 07 |
| DPO: reward ↑ and answers 2× longer | length exploitation | `sigmoid_norm`/SimPO-like loss, length-balanced pairs, higher β | 10 |
| DPO: `logps/chosen` collapses | likelihood displacement | lower LR, `apo_zero`, add SFT term | 10 |
| GRPO: reward flat, `frac_reward_zero_std` ≈ 1 | tasks too easy or too hard | change data difficulty; larger G; higher temperature | 12 |
| GRPO: lengths explode | reward doesn't penalize rambling; clipped completions | length penalty; raise `max_completion_length`; Dr. GRPO-style loss | 12 |
| `target modules not found` | module names differ for this architecture | `print(model)`; use `all-linear` | 05 |
| MoE: 4-bit didn't save memory | fused experts aren't `nn.Linear` | 16-bit + small MoE, or offload; know the limitation | 13 |
| MoE error about `num_experts` | config naming differences across families | patched in `modeling.load_model` | 13 |
| very slow on T4 | QLoRA dequantization + checkpointing + long seq | shorter `max_length`, LoRA 16-bit if it fits, Unsloth | 04 |

## 6. Glossary

**Adapter**: small trained weights (LoRA) added to a frozen model. **Assistant-only loss**: loss on assistant turns
only. **Base model**: pre-trained only. **BF16/FP16**: 16-bit float formats (range vs precision). **Chat template**:
Jinja program that flattens messages into tokens. **Completion-only loss**: loss on the completion of a
prompt/completion pair. **DPO**: preference optimization via implicit rewards. **Effective batch**: per-device ×
accumulation × GPUs. **EOS**: end-of-sequence/turn token. **GRPO**: RL with group-normalized advantages. **KL
penalty**: keeps the policy near a reference. **LoRA**: low-rank update `(α/r)·BA`. **MoE**: model with routed expert
MLPs; total vs active params. **NF4**: 4-bit NormalFloat quantization. **Packing**: concatenating examples into
full-length rows. **PEFT**: parameter-efficient fine-tuning. **QLoRA**: LoRA on a 4-bit base. **Reference model**:
frozen policy the KL/DPO term compares against. **Reward model**: scores responses (Bradley–Terry). **RLVR**: RL
with verifiable (programmatic) rewards. **SFT**: supervised fine-tuning on demonstrations.

## 7. Reading list (in course order)

- Vaswani et al., *Attention Is All You Need* (2017)
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020); Hoffmann et al., *Chinchilla* (2022)
- Penedo et al., *The FineWeb Datasets* (2024)
- Ouyang et al., *InstructGPT* (2022); Zhou et al., *LIMA* (2023)
- Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023); Biderman et al., *LoRA Learns Less and Forgets Less* (2024);
  Thinking Machines, *LoRA Without Regret* (2025)
- Rajbhandari et al., *ZeRO* (2020); Dao et al., *FlashAttention* (2022)
- Rafailov et al., *DPO* (2023); Azar et al., *IPO* (2023); Ethayarajh et al., *KTO* (2024); Hong et al., *ORPO* (2024);
  Meng et al., *SimPO* (2024); Tunstall et al., *Zephyr* (2023)
- Shao et al., *DeepSeekMath / GRPO* (2024); DeepSeek-AI, *DeepSeek-R1* (2025); Yu et al., *DAPO* (2025);
  Liu et al., *Understanding R1-Zero-Like Training (Dr. GRPO)* (2025)
- Shazeer et al., *Sparsely-Gated MoE* (2017); Fedus et al., *Switch Transformer* (2021); Zoph et al., *ST-MoE* (2022);
  Jiang et al., *Mixtral* (2024); Dai et al., *DeepSeekMoE* (2024); Muennighoff et al., *OLMoE* (2024)
- Qi et al., *Fine-tuning Aligned Language Models Compromises Safety* (2023)
- Lambert et al., *Tülu 3* (2024) and the SmolLM3 blog (2025): two fully open post-training recipes

## 8. Where to go next (portfolio ideas)

1. **Your own data:** pick a real task from your work (e.g. ticket → structured fields), build 1–2k examples, and run the full recipe.
2. **Function calling / tool use:** SFT a small model on tool-call conversations (the `tools` argument of chat templates), evaluated by executing the calls.
3. **Distillation:** generate answers with a big model for your task, then SFT a 0.6B student; compare cost and quality.
4. **Multilingual:** fine-tune for an Indian language (e.g. Hindi instruction data), and measure tokenizer efficiency first.
5. **Reward model:** train one with TRL's `RewardTrainer` on preference pairs, then use it in `evaluate`.
6. **One-stage alignment:** ORPO vs SFT→DPO on the same data and compute.
7. **Speed:** re-run Project 1 with Unsloth on the T4 and compare minutes and memory.
8. **Vision-language:** LoRA on a small VLM (e.g. SmolVLM) for document extraction. The same ideas carry over.

**You made it.** Fill in the README's results table with your own runs. That table, the configs and the model
cards on your Hub profile are a stronger portfolio than any certificate.
