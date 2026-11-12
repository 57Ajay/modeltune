# 10 · Preference optimization: from imitation to judgment (RLHF, DPO and friends)

> **Why this chapter:** after SFT a model can answer, but "an answer" isn't "the answer people prefer". Preference
> optimization teaches the model to rank its own possible outputs the way humans (or a strong judge) would. This is
> the theory behind Project 2's second stage, and the conceptual bridge to RL in chapter 12.

## 1. Why SFT isn't enough

- **Imitation ceiling:** SFT can only reproduce its demonstrations, including their flaws.
- **No negatives:** SFT never sees "this answer is *worse*". It can't learn to avoid a failure it was never shown.
- **Judging is easier than writing:** a labeler can reliably say "B is better than A" for answers they couldn't
  have written themselves. Preference data is cheaper and scales to subtle qualities (helpfulness, tone,
  correctness of long reasoning, safety).

The data: triples `(x, y_w, y_l)`, a prompt with a **chosen** (winner) and a **rejected** (loser) response. Sources
are human labelers, or **AI feedback** (RLAIF): a strong model rates candidate answers. **UltraFeedback**, which we
use, has ~64k prompts, each with 4 responses from different models rated by GPT-4. The "binarized" version takes
the best as chosen and a lower-rated one as rejected.

## 2. Modeling preferences: Bradley–Terry

Assume each response has a latent quality r(x, y), and humans prefer y_w with probability

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )          σ = logistic sigmoid
```

A **reward model** is an LLM with a scalar head trained to maximize this likelihood:
`L_RM = −log σ(r_w − r_l)`. That's what `Skywork-Reward-V2-Qwen3-0.6B` (our evaluator) is.

## 3. Classic RLHF: reward model + PPO

Optimize the policy π to get high reward **without drifting too far** from the SFT model π_ref:

```
max_π  E_{x, y∼π} [ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

The KL term matters: without it the policy finds outputs the reward model over-scores (**reward hacking**:
gibberish, sycophancy, extreme length) and loses fluency. PPO optimizes this online: sample from π, score with the
reward model, estimate advantages with a learned **value model**, take clipped policy-gradient steps. That's
four models in memory (policy, reference, reward, value) plus generation in the loop. It's powerful (InstructGPT,
early ChatGPT) but heavy and finicky.

## 4. DPO: the same objective, no RL

Rafailov et al. (2023), *"Your Language Model is Secretly a Reward Model"*. The KL-constrained objective above has a
closed-form optimum:

```
π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x)
```

Rearranged, any reward can be written in terms of its optimal policy:
`r(x,y) = β log [π*(y|x) / π_ref(y|x)] + β log Z(x)`. Substituting into Bradley–Terry, the intractable `Z(x)`
**cancels** (it's the same for both responses), leaving a loss on the policy directly:

```
L_DPO = −E [ log σ( β · log πθ(y_w|x)/π_ref(y_w|x)  −  β · log πθ(y_l|x)/π_ref(y_l|x) ) ]
                    └──────── implicit reward r̂_w ─────┘   └──────── implicit reward r̂_l ─────┘
```

What that buys you: no reward model, no sampling, no value model. It's a classification-style loss on fixed
pairs: two forward passes through the policy and two through the reference per pair.

**Gradient intuition:** raise log π(y_w), lower log π(y_l), weighted by `σ(r̂_l − r̂_w)`, i.e. *more* when the
model currently ranks the pair wrongly, almost nothing when it's already confident and right.

**β:** the strength of the leash to π_ref. Small β (0.01–0.05) allows big deviations; large β (0.5) stays close.
TRL's default 0.1 is a good start.

**The reference model** is normally the SFT model. In Project 2 we start DPO from the SFT **adapter**, and TRL
keeps a frozen copy of it (adapter `ref`) as π_ref, so no second model is loaded. (That copy is why a `ref/`
folder appears next to the trained adapter.)

## 5. Reading DPO logs

| Log | Meaning | Healthy trend |
|---|---|---|
| `loss` | starts at **ln 2 ≈ 0.693** (policy = reference, so r̂_w = r̂_l = 0 and σ(0) = ½) | decreases (often to 0.5–0.6 on noisy data; near 0 means overfitting) |
| `rewards/chosen`, `rewards/rejected` | mean implicit rewards β·log(π/π_ref) | rejected falls; chosen ideally flat or rising |
| `rewards/margins` | chosen − rejected | grows |
| `rewards/accuracies` | fraction of pairs with r̂_w > r̂_l | rises from 0.5 toward 0.6–0.8 |
| `logps/chosen`, `logps/rejected` | raw log-probs under the policy | watch for *both* falling (see below) |

(Our CPU smoke test logs `loss: 0.6931` at step 1, as the theory says.)

## 6. Failure modes

1. **Likelihood displacement:** the margin grows mostly by pushing *both* responses down, the rejected one faster.
   The chosen answer's probability can end up lower than before DPO, and probability mass leaks to unseen,
   sometimes worse, outputs. Mitigations: lower LR and fewer steps; add an SFT term on chosen (`loss_type:
   [sigmoid, sft]` with `loss_weights`, the "RPO" idea); **APO-zero** (`loss_type: apo_zero`) anchors the chosen
   likelihood to go up.
2. **Length exploitation:** chosen responses are on average longer in most preference datasets, so DPO learns
   "longer is better". Watch `avg_answer_words` in `modeltune.evaluate`. Mitigations: length-normalized
   objectives (SimPO, `sigmoid_norm`), length-balanced data.
3. **Over-optimization:** too many epochs or a high LR give repetitive, degenerate or sycophantic text. One epoch,
   LR around 5e-7 (full FT) or 5e-6 – 5e-5 (LoRA).
4. **Off-policy data:** UltraFeedback's responses came from *other* models. Pairs sampled from *your* SFT model and
   then judged ("on-policy") usually work better. Iterating (sample → judge → DPO → repeat) is **online/iterative
   DPO**, a step toward RL.

## 7. The family (most available as a `loss_type` or trainer in TRL)

| Method | Key idea | Needs reference? | Data |
|---|---|---|---|
| **DPO** (`sigmoid`) | Bradley–Terry on implicit rewards | yes | pairs |
| **IPO** (`ipo`) | squared loss toward a fixed margin; resists overfitting deterministic preferences | yes | pairs |
| **cDPO / robust** (`label_smoothing`, `robust`) | assume some labels are flipped | yes | noisy pairs |
| **APO-zero / down** (`apo_zero`, `apo_down`) | anchor whether chosen goes up or down in absolute terms; used for SmolLM3 | yes | pairs |
| **KTO** (`KTOTrainer`) | prospect-theory loss on single examples marked good/bad | yes | thumbs up/down, unpaired |
| **ORPO** | SFT loss + odds-ratio penalty in *one* stage | **no** | pairs |
| **SimPO** | length-normalized average log-prob + margin | **no** | pairs |
| **Online DPO / RLOO / PPO / GRPO** | generate on-policy during training, score with RM or rules | varies | prompts (+ RM or checker) |

Rule of thumb: start with DPO (sigmoid) on top of a good SFT model. If outputs balloon in length, try a
length-normalized or anchored variant. If you only have thumbs-up/down logs, use KTO. If you want one stage
instead of SFT then DPO, try ORPO.

## 8. Where it lives in the code

`configs/p2_smollm3_dpo_qlora.yaml` sets `method: dpo` and `model.adapter: <SFT adapter>`. `modeltune.train` loads the
4-bit base, attaches the SFT adapter as **trainable**, passes no new `peft_config`, and TRL's `DPOTrainer` creates
the frozen `ref` copy itself. The data (`trl-lib/ultrafeedback_binarized`) is already in TRL's preference format
with an implicit prompt, so no formatter is needed. Any `DPOConfig` field (`beta`, `loss_type`, `max_length`, …)
goes in the `train:` block.

## Check yourself
1. Derive why `Z(x)` cancels in the DPO loss. Why couldn't we just optimize the RLHF objective directly with it present?
2. DPO's loss falls nicely but `logps/chosen` dropped by 30 nats. Is the model better? What do you check?
3. Why does the DPO loss start at exactly ln 2?
4. You only have production logs with 👍/👎 on single responses. Which method fits?
5. What does β control, and what do you expect from β = 0.01 vs β = 0.5?

**Next:** [11 · Project 2: base → aligned](11_project2_base_to_aligned.md)
