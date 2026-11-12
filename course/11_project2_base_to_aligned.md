# 11 · Project 2: SmolLM3-3B-Base → chat assistant (SFT) → aligned assistant (DPO), with QLoRA

> **What you'll build:** a mini version of a real post-training pipeline. You take a raw base model that can only
> continue documents, teach it to hold a conversation (SFT on multi-turn chats), then teach it which answers people
> prefer (DPO on preference pairs), all in 4-bit on a free T4.
> **What you'll learn:** making a base model chat-ready (templates, EOS, pad), QLoRA in practice, assistant-only
> loss on multi-turn data, chaining training stages through adapters, and evaluating open-ended quality.
>
> Notebook: [`notebooks/02_project2_sft_dpo_qlora.ipynb`](../notebooks/02_project2_sft_dpo_qlora.ipynb) ·
> Configs: [`p2_smollm3_sft_qlora.yaml`](../configs/p2_smollm3_sft_qlora.yaml),
> [`p2_smollm3_dpo_qlora.yaml`](../configs/p2_smollm3_dpo_qlora.yaml) ·
> Runtime: roughly 45–70 min (SFT) + 50–80 min (DPO) on a T4 (estimates; lower `data.max_train_samples` if your session is short)

## 1. Why this setup

| Choice | Reason |
|---|---|
| **SmolLM3-3B-Base** | fully open (weights, data mixture and training recipe published), Apache-2.0, a real 3B model (36 layers, GQA 16:4, 64k context). Starting from **Base** lets you see what post-training adds |
| **QLoRA** | 3.08B × 2 B = 6.2 GB of 16-bit weights plus activations is tight on 15 GB; NF4 brings weights to ~2 GB (chapter 04 §5 has the full estimate) |
| **Two stages** | SFT teaches *how to answer*; DPO teaches *which answer is better*. DPO on a model that can't chat yet would have nothing sensible to rank |
| **Same evaluation for all three models** | 100 UltraFeedback test prompts, greedy decoding, scored by the same reward model, so base, SFT and DPO numbers are comparable |

## 2. Stage 0: meet the base model

```bash
python -m modeltune.chat configs/p2_smollm3_sft_qlora.yaml --base --prompt "Give me three tips for writing clean Python code."
python -m modeltune.evaluate configs/p2_smollm3_sft_qlora.yaml --base
```

Typical base-model behaviour: it may answer briefly, drift into unrelated text, write the *next user turn itself*,
or run to `max_new_tokens` without stopping. It has knowledge, but no idea that `<|im_end|>` after an assistant turn
means *stop*. Note the reward score and `avg_answer_words`.

### Making a base model chat-ready (what the config does)
```yaml
model:
  id: HuggingFaceTB/SmolLM3-3B-Base
  quantization: 4bit
  chat_template: templates/chatml.jinja   # the base tokenizer has NO template
  eos_token: "<|im_end|>"                 # stop at end of the assistant TURN, not end of document
```
- **Template:** ChatML, because SmolLM3's vocabulary already contains `<|im_start|>` and `<|im_end|>` with
  normal-looking embeddings, so no new untrained tokens are needed (chapter 06).
- **EOS:** the base EOS `<|end_of_text|>` means "end of document". We make `<|im_end|>` the EOS so that training
  teaches it as "end of my turn" and `generate()` stops on it (`modeling.load_model` syncs `generation_config`).
- **Pad:** the base has none; we reuse EOS. That's safe because TRL masks padding by attention mask.
- The template wraps assistant content in `{% generation %}` tags, which is what makes `assistant_only_loss` possible.

## 3. Stage 1: SFT on multi-turn conversations

Data: **`trl-lib/Capybara`**: ~16k high-quality multi-turn conversations in `messages` format, so no formatter is
needed. We use 2,000 of them.

```yaml
lora: {r: 32, alpha: 32, dropout: 0.05, target_modules: all-linear}   # ≈60M trainable: a broad new behavior needs more capacity than P1
train:
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8      # effective batch 16
  gradient_checkpointing: true        # the big memory saver (chapter 04)
  max_length: 1024
  assistant_only_loss: true           # loss on every assistant turn, none on user turns
  optim: paged_adamw_8bit             # 8-bit Adam states, paged to CPU on spikes
```

What to check in the training output:
1. **The tokenized example:** the loss block should contain *only* assistant text, each piece ending in
   `<|im_end|>`. (Our CPU smoke test showed 773 of 892 tokens in the loss for one long Capybara chat, all assistant
   turns.) If user text appears there, the template's generation tags aren't being used.
2. **`trainable params`** ≈ 60M (about 2% of the model).
3. **Loss** falls from the base model's starting point and flattens; `eval_loss` tracks it.
4. **Peak memory** around the chapter 04 estimate (~7 GB).

Then evaluate and chat:
```bash
python -m modeltune.evaluate configs/p2_smollm3_sft_qlora.yaml
python -m modeltune.chat configs/p2_smollm3_sft_qlora.yaml --prompt "Give me three tips for writing clean Python code."
```
Expect the biggest jump of the whole project here: the model now answers the question, in assistant style, and
**stops**. The reward score should rise substantially over the base model.

## 4. Stage 2: DPO on preference pairs

Data: **`trl-lib/ultrafeedback_binarized`**: chosen/rejected conversations with an implicit prompt, already in
TRL's preference format. We use 2,000 pairs.

```yaml
method: dpo
model:
  adapter: 57Ajay/smollm3-3b-sft-qlora   # start from the SFT adapter (pushed by stage 1)
train:
  learning_rate: 1.0e-5                 # ~20× lower than SFT: nudge, don't rewrite
  beta: 0.1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  max_length: 1024
```

How the pieces fit:
- `modeltune.train` loads the 4-bit base and attaches the **SFT adapter as trainable**. Because `model.adapter` is
  set, no new LoRA is created.
- TRL's `DPOTrainer` sees a PEFT model without a `ref_model` and **copies the adapter into a frozen `ref`
  adapter**. The reference policy is exactly your SFT model at zero extra memory. (That copy is saved in a
  `ref/` subfolder next to the trained adapter. The adapter at the top level is the DPO result.)
- Each step runs the policy and reference over chosen and rejected responses: about 2× the forward cost of SFT
  per example, which is why DPO is slow on a T4.

Watch (chapter 10 §5): `loss` starting at ≈0.693, `rewards/accuracies` rising above 0.5, `rewards/margins` growing,
and `logps/chosen` not collapsing.

## 5. Compare base → SFT → DPO

The notebook builds a table from the three `eval_*.json` files:

| | reward_mean | avg_answer_words |
|---|---|---|
| base (with ChatML) | ? | ? |
| SFT | ? | ? |
| SFT + DPO | ? | ? |

How to read it:
- **base → SFT** should be the largest gain: format and helpfulness.
- **SFT → DPO** is usually a smaller gain with 2k pairs. That's normal; real pipelines use 10–100× more preference data.
- If DPO's reward rises **together with a big length increase**, read the answers before celebrating: the reward
  model may just like longer text (chapter 08 §6, chapter 10 §6).
- If DPO *lowers* the score: check `logps/chosen` (likelihood displacement), and try a lower LR, higher β, or
  `loss_type: apo_zero`.
- **Read 10 answers from each model side by side.** Numbers on 100 prompts with a 0.6B judge are a rough guide, not a verdict.

## 6. Practical notes for Colab

- Each stage pushes its adapter to the Hub when it finishes, so a disconnect between stages costs nothing. To use
  the local SFT adapter instead: `model.adapter=outputs/p2-smollm3-3b-sft-qlora`.
- For long stages use checkpoints: `train.save_strategy=steps train.save_steps=50`, and resume with
  `train.resume_from_checkpoint=true`.
- Short on time? Use `data.max_train_samples=800` for each stage. The *direction* of the results usually survives.

## 7. What transferred from Project 1, and what was new

| From Project 1, unchanged | New in Project 2 |
|---|---|
| config-driven loop, `evaluate --base` first, reading the tokenized example | base model: template, EOS and pad decisions are **yours** |
| LoRA mechanics and LR heuristics | **QLoRA**: NF4 weights, compute dtype, paged 8-bit optimizer |
| `chat`, `merge`, Hub pushes | multi-turn **assistant-only** loss via generation tags |
| error analysis by reading outputs | **chaining stages**: an adapter as the starting point and the reference |
| | open-ended evaluation with a **reward model**, plus length as a sanity metric |

## 8. Experiments

| Experiment | How | Lesson |
|---|---|---|
| DPO without SFT | DPO config with `model.adapter=null` (fresh LoRA; reference = raw base) | why SFT comes first |
| β sweep | `train.beta=0.01` / `0.5` | the leash to the reference; watch length |
| Anchored / normalized loss | `train.loss_type=apo_zero` or `sigmoid_norm` | fixing likelihood displacement / length bias |
| More SFT data | `data.max_train_samples=6000` (SFT) | where the quality actually comes from |
| Merge then serve | `python -m modeltune.merge configs/p2_smollm3_dpo_qlora.yaml --push` | merging into the 16-bit base (chapter 14). Note that the DPO adapter was trained on top of SFT, so its weights already include the SFT update |

## Check yourself
1. Why can't we just keep the base model's `<|end_of_text|>` as EOS for chat?
2. In DPO, what exactly is the reference model here, and how much extra GPU memory does it cost?
3. Why is the SFT stage given rank 32 but Project 1 only rank 16?
4. Your DPO model scores +1.5 over SFT and writes 2.3× more words. What do you conclude, and what do you try next?

**Next:** [12 · RL & reasoning with GRPO](12_rl_grpo_reasoning.md)
