# 12 · RL & reasoning: GRPO with verifiable rewards

> **Why this chapter:** SFT imitates and DPO ranks fixed pairs; reinforcement learning lets the model **learn from
> the consequences of its own outputs**. With rewards a program can check (is the math answer right? do the tests
> pass?), RL is what turned LLMs into today's reasoning models. GRPO is the algorithm that made it cheap.
>
> Notebook: [`notebooks/03_bonus_grpo_gsm8k.ipynb`](../notebooks/03_bonus_grpo_gsm8k.ipynb) ·
> Config: [`configs/bonus_grpo_qwen3_gsm8k.yaml`](../configs/bonus_grpo_qwen3_gsm8k.yaml) ·
> Runtime: ~40–60 min on a T4 (estimate)

## 1. RL vocabulary, mapped to LLMs

| RL term | For an LLM |
|---|---|
| policy π | the model |
| state | prompt + tokens generated so far |
| action | the next token |
| episode / trajectory | one full completion |
| reward | a scalar for the completion (usually only at the end) |
| on-policy | training data is sampled from the *current* model |

The difference from SFT: there is no target text. The model generates, gets scored, and is pushed toward
higher-scoring behavior. It can discover solutions nobody wrote down, and it learns from its *own* distribution,
which avoids the mismatch between training on someone else's text and generating your own.

## 2. Policy gradients in one equation

```
∇J(θ) = E_{y∼π} [ A(x, y) · ∇ log π_θ(y | x) ]        A = advantage = how much better than expected
```

Raise the log-probability of completions that did better than expected, lower it for those that did worse.
Using raw reward as A works (REINFORCE) but is very noisy. Subtracting a **baseline** (the expected reward)
reduces variance without bias. **PPO** learns that baseline with a separate **value model** and clips the update
ratio `π_θ/π_old` to [1−ε, 1+ε] so each step stays small. For LLMs that means policy + value + reference + reward
model in memory, which is the cost GRPO removes.

## 3. GRPO: the group is the baseline

**Group Relative Policy Optimization** (Shao et al., *DeepSeekMath*, 2024):

```
for each prompt x:
    sample G completions y₁ … y_G from π_old            (G = num_generations, e.g. 8)
    score them: r₁ … r_G                                (reward functions)
    A_i = (r_i − mean(r)) / std(r)                      (same advantage for every token of y_i)
loss = − mean over tokens of  min( ρ·A, clip(ρ, 1−ε, 1+ε)·A )   [+ β·KL(π_θ ‖ π_ref)]
       with ρ = π_θ(token) / π_old(token)
```

- **No value model:** the group mean *is* the baseline. "Was this answer better than my other attempts at the same
  question?" That's a large memory and complexity win.
- **Zero-variance groups teach nothing:** if all 8 answers are right (or all wrong), every A_i = 0. RL works best
  on problems the model solves **sometimes**. A model at 0% can't start (no signal), and at 100% has nothing to
  learn. Curriculum and data difficulty matter.
- **KL to a reference:** originally included (β > 0). Recent practice (DAPO, Dr. GRPO) often drops it; TRL's
  default is now `beta: 0.0`, which also saves the reference forward pass.

### Variants you'll see (TRL `loss_type` and friends)
| Variant | Change | Why |
|---|---|---|
| **Dr. GRPO** (Liu et al., 2025) | remove per-sequence length normalization and std scaling | those biases reward long wrong answers |
| **DAPO** (ByteDance, 2025) | clip-higher (ε_high > ε), token-level loss averaging, dynamic sampling (drop zero-variance groups), overlong penalty | stability and entropy at scale; TRL's default `loss_type` is `dapo` |
| **GSPO** (Qwen, 2025) | importance ratio per *sequence* instead of per token | more stable, especially for MoE models |
| **RLOO** | leave-one-out mean as baseline | close cousin; `RLOOTrainer` |

## 4. Rewards: the whole game

**Verifiable rewards** (RLVR) are computed by code: exact numeric match, unit tests passing, JSON validating against
a schema, a regex for format. They're cheap, objective, and hard to fool (but not impossible). Ours
(`modeltune/rewards.py`):

```python
correctness(...)   # 2.0 if the number in <answer>…</answer> equals GSM8K's "#### N", else 0.0
format_reward(...) # 0.5 if the WHOLE completion is <reasoning>…</reasoning><answer>…</answer>
```

Design principles:
- **The real objective must dominate.** Format is a small shaping bonus (0.5 vs 2.0); otherwise the model
  "learns" to emit perfect tags around wrong answers.
- **Assume the model will exploit your checker.** If the parser took *any* number in the text, the model could list
  many candidates. Ours takes the last number inside `<answer>`, and the strict format reward discourages
  wandering text.
- **Learned reward models and LLM judges** extend RL to unverifiable tasks (helpfulness, writing), but are much
  easier to hack (chapter 10). Keep a KL leash and evaluate with a *different* judge.

## 5. Reasoning models, briefly

- OpenAI's o1 (2024) showed that RL-trained long chains of thought scale with **test-time compute**: thinking
  longer gives better answers.
- **DeepSeek-R1** (2025) published the recipe: GRPO with rule-based rewards on math and code. **R1-Zero**, RL directly
  on a base model with no SFT, developed long reasoning, self-verification and "aha" moments on its own. Reasoning
  was then **distilled** into small models by SFT on R1's traces.
- Open debate: does RLVR teach new abilities or mostly *sharpen* what the base model can already sometimes do?
  Yue et al. (2025) found base models match or beat their RL-tuned versions at pass@k for large k, suggesting
  RL mainly improves reliability (pass@1). Practically: **RL amplifies a capable base**, which is why it's
  applied after pre-training and SFT, not instead of them.

## 6. Our config, and what to watch

```yaml
model: {id: Qwen/Qwen3-0.6B}
lora: {r: 16, alpha: 32, dropout: 0.0, target_modules: all-linear}   # LoRA works very well for RL (chapter 05)
data: {dataset: openai/gsm8k, subset: main, formatter: modeltune.formatters:gsm8k}   # prompt + "answer" column
rewards: [modeltune.rewards:correctness, modeltune.rewards:format_reward]
train:
  max_steps: 100
  learning_rate: 2.0e-5
  num_generations: 8                  # G
  per_device_train_batch_size: 8      # counted in COMPLETIONS: 8 = one prompt's group
  gradient_accumulation_steps: 2      # 2 prompts per optimizer step
  max_completion_length: 256
  temperature: 1.0                    # diversity within a group = learning signal
  chat_template_kwargs: {enable_thinking: false}   # our own <reasoning> tags instead of Qwen's <think>
```

| Log | Meaning | Watch for |
|---|---|---|
| `reward` (and per-function `rewards/...`) | mean total reward | rising; format usually rises first |
| `reward_std` | spread within groups | must stay > 0; ~0 means no learning signal |
| `frac_reward_zero_std` | fraction of groups where all G answers scored the same | the share of prompts teaching nothing; if it nears 1, the task is too easy or too hard |
| `completions/mean_length` | answer length | sudden growth = rambling; hitting the max = truncation |
| `completions/clipped_ratio` | completions cut at `max_completion_length` | high means raise the limit or penalize length |
| `entropy` | policy randomness | collapsing to ~0 means exploration has died |

**Cost reality:** generation dominates. Every step samples 16 completions of up to 256 tokens with plain
`model.generate` on a T4. On bigger GPUs, TRL can generate with **vLLM** (`use_vllm: true`), which is many times
faster.

Evaluate before and after with `metric: gsm8k` (accuracy and format rate, greedy decoding):
```bash
python -m modeltune.evaluate configs/bonus_grpo_qwen3_gsm8k.yaml --base
python -m modeltune.train    configs/bonus_grpo_qwen3_gsm8k.yaml
python -m modeltune.evaluate configs/bonus_grpo_qwen3_gsm8k.yaml
```
With 100 steps, expect the **format rate** to improve a lot and accuracy to improve modestly. This run is about
seeing the mechanics, not setting records.

## 7. Experiments

| Experiment | Override | Lesson |
|---|---|---|
| No format reward | `rewards=[modeltune.rewards:correctness]` | does format survive without shaping? |
| Group size | `train.num_generations=4 train.per_device_train_batch_size=4` | fewer samples means noisier advantages |
| Low temperature | `train.temperature=0.6` | less diversity means more zero-variance groups |
| KL on | `train.beta=0.04` | stability vs speed of change |
| Your own reward | add a function to `rewards.py`, list it in `rewards:` | e.g. penalize answers > 150 words |

## Check yourself
1. Why does GRPO not need a value model? What plays its role?
2. A batch has 8 prompts; for 6 of them, all 8 completions got the same reward. How much did those 6 contribute to the gradient?
3. Why is the format reward 0.5 and correctness 2.0, not the other way around?
4. Why does RL need a base model that's already sometimes right?
5. What would you change to run this on an A100 much faster?

**Next:** [13 · Mixture of Experts](13_mixture_of_experts.md)
