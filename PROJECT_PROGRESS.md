# RocketRL — Project Progress & Experimental Log

**Project:** Autonomous Rocket Landing with Deep Reinforcement Learning  
**Last updated:** 26 October 2026  
**Current stage:** Gymnasium/Box2D LunarLander experiments completed through three ~1-million-step PPO policies; next stage is trajectory diagnostics and reward-objective design.  
**Status:** Research/portfolio prototype, not a real rocket guidance system.

> **Purpose of this file:** Preserve what was implemented, the observed results, the interpretation of each experiment, and the decision that followed it. This is a retrospective and decision log, **not** a claim that every experiment achieved its intended objective.

---

## 1. Project goal and scope

The long-term goal of **RocketRL** is to investigate how reinforcement learning can control a simulated rocket during landing. The initial focus is deliberately **2D LunarLander-v3**, so that PPO, evaluation, and reward design can be understood before constructing a custom 3D rocket environment.

**Current research question:**

> How do training duration and reward design influence safe resting, landing-pad accuracy, and strict landing success in a reinforcement-learning lander?

**Scope implemented so far:**

- Run the open-source Gymnasium `LunarLander-v3` environment using Box2D physics.
- Establish an untrained random-action reference and view the environment.
- Train a standard PPO agent, then continue its training.
- Build a repeatable 100-episode evaluation with physical-state checks.
- Investigate the gap between high Gymnasium rewards and precise physical landings.
- Train and evaluate two alternative reward designs: terminal precision reward and potential-based shaping.

**Not implemented yet:** A custom 3D Godot rocket, Godot RL Agents integration, fuel-constrained custom physics, wind/parameter randomization, or a multi-seed training study. Those are possible future extensions, **not** completed work.

## 2. Technology and development setup

| Component | Implementation | Role |
|---|---|---|
| Editor | Visual Studio Code | Python development and debugging |
| Version control | Git/GitHub repository `RocketRL` | Track project changes |
| Language | Python 3.12 | Training and evaluation scripts |
| Environment API | Gymnasium (`LunarLander-v3`) | Reset/step/observation/action interface |
| Physics | Box2D | Gravity, motion, collision/contact simulation |
| RL library | Stable-Baselines3 | PPO implementation and model serialization |
| Deep learning | PyTorch | Policy/value neural networks |
| Experiment visualization | TensorBoard | Training reward, length, and optimizer logs |
| Statistics | Python `statistics` module | Evaluation mean and sample standard deviation |

The initial installation log showed Gymnasium **1.4.0**, Stable-Baselines3 **2.9.0**, Box2D **2.3.10**, and PyTorch **2.14.1**. The project uses a local `.venv` and a generated `requirements.txt`.

**Working directory on the development machine:** `C:\ResumeProjects\RocketRL`

### Setup used

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "gymnasium[box2d]" "stable-baselines3[extra]"
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

The first attempt to visualize the random agent did not display a window; later testing confirmed that human-mode graphical rendering worked. No enduring environment installation issue was reported.

### Code and artifact organization

The following is the project structure discussed and used during development. **Confirm local file presence before treating this as an exact repository inventory**; some diagnostic scripts may have been temporary.

```text
RocketRL/
├── lunar_lander/
│   ├── random_agent.py          # Random-action reference
│   ├── train.py                 # Original PPO, first ~300k steps
│   ├── continue_training.py     # Continue original PPO to ~1M total
│   ├── evaluate.py              # Render trained agent (five episodes)
│   ├── evaluate_metrics.py      # Common 100-episode evaluation
│   ├── rewards.py               # PrecisionLandingReward, PotentialBasedReward
│   ├── train_precision.py       # Terminal precision-reward experiment
│   ├── train_potential.py       # Potential-based shaping experiment
│   └── test_potential.py        # Proposed wrapper smoke test, if created
├── models/                     # Saved PPO checkpoints
├── runs/                       # TensorBoard logs
├── results/                    # Experimental tables and later trajectory logs
├── README.md
├── requirements.txt
├── .gitignore
└── PROJECT_PROGRESS.md         # This file
```

Key model filenames used in the workflow:

```text
models/ppo_lunar_lander.zip       # ~300k-step baseline
models/ppo_lunar_lander_1m.zip    # Continued standard PPO
models/ppo_precision_1m.zip       # New PPO trained with terminal precision reward
models/ppo_potential_1m.zip       # New PPO trained with potential-based shaping
```

> These are the filenames configured in the training scripts. Confirm checkpoints exist locally before attempting to load them. Avoid overwriting them when creating further experiments.

---

## 3. Environment and learning design

### LunarLander interface

The original experiments use `gym.make("LunarLander-v3")` with the **discrete** action space.

- **Eight observations:** horizontal and vertical position; horizontal and vertical velocity; angle; angular velocity; contact flag for each landing leg.
- **Four discrete actions:** no engine; left orientation engine; main engine; right orientation engine.
- **Episode:** starts with `env.reset()` and advances through calls to `env.step(action)` until `terminated` or `truncated`.
- **Random agent:** `env.action_space.sample()` chooses actions without using the observation.
- **Trained agent:** `model.predict(obs, deterministic=True)` chooses the learned policy's preferred action during evaluation; the second return value is unused recurrent state (`None` for this nonrecurrent MLP policy).

Training normally runs without `render_mode="human"` for speed. Short visual evaluations can use human rendering; quantitative evaluations run without rendering.

### PPO setup shared by fresh training runs

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="runs/",
    seed=42,
    device="cpu",
)
```

- `MlpPolicy`: multilayer-perceptron policy for numerical observations; PPO also learns a value function.
- `n_steps=2048`: rollout steps collected before a PPO optimization cycle (one training environment).
- `gamma=0.99`: discount factor for future reward.
- `seed=42`: fixed **training** random seed for the documented fresh runs; one seed does not demonstrate reproducibility across runs.
- `Monitor(...)`: wrapper around training environments to log episode statistics.

The original model was continued using `PPO.load(...)`, not restarted. The two alternative-reward policies were initialized as fresh models.

### Model and evaluation principles

1. **Training reward is not an unbiased evaluation metric**, especially when the reward function differs between experiments.
2. All quantitative comparisons below were run on the **unmodified** `gym.make("LunarLander-v3")` with `deterministic=True`.
3. Every reported 100-episode evaluation used episode seeds `1000, 1001, ..., 1099`.
4. A single trained policy evaluated across 100 episodes estimates performance on those initial conditions; it does not replace repeated training with different random seeds.
5. Gymnasium's `reward >= 200` per episode is a *high-score count*, not the same thing as the usual solved benchmark (mean return >= 200) and not equivalent to precise physical landing.

---

## 4. Evaluation methodology

### Episode-level quantities

For every episode, `evaluate_metrics.py` records:

- **Total original reward**: sum of step rewards.
- **Episode length**: simulation steps until ending.
- **High-score episode**: total reward >= 200.
- **Final physical state**: examined through `env.unwrapped` to determine safe rest, center-on-pad position, and leg contact.

Core classification logic:

```python
safe_rest = (
    terminated
    and not truncated
    and not sim.game_over
    and not sim.lander.awake
)

on_pad = (
    sim.helipad_x1 <= sim.lander.position.x <= sim.helipad_x2
)

both_legs = all(leg.ground_contact for leg in sim.legs)

strict_landing = safe_rest and on_pad and both_legs
```

### Reported categories

**Overlapping diagnostic counts** (all restricted to safe-rest episodes):

- `safe_rest_landings`: all safe rests.
- `safe_rest_on_pad`: safe rests with lander **center** horizontally inside the pad bounds.
- `safe_rest_both_legs`: safe rests with both legs reporting terrain contact.

**Mutually exclusive outcome categories** (sum to 100):

1. **Strict landing**: safe rest, center on pad, both legs reporting contact.
2. **Off-pad safe rest**: safe rest but center outside pad boundaries.
3. **On-pad safe rest with insufficient leg contact**: safe rest, center on pad, not both legs reporting contact.
4. **No safe rest**: anything failing `safe_rest` (e.g., crashes, out-of-bounds termination, or truncation). **Do not equate this category with crashes alone.**

The final script added assertions to check that the exclusive categories sum to the episode count and are consistent with overlapping diagnostic counts.

### Metric caveats

- The strict definition is **custom**, not Gymnasium's standard "solved" criterion.
- `on_pad` tests the lander body's **center**, not the actual coordinates of both landing feet.
- `ground_contact` indicates collision contact with terrain, not necessarily contact specifically with the platform.
- `env.unwrapped` uses internal implementation details that may differ between Gymnasium releases.
- The reported sample standard deviation uses Python's `statistics.stdev`.
- Only summary metrics were retained in this record; individual episode data/trajectories have not yet been consolidated into a research dataset.

---

## 5. Chronological implementation and experiment record

### E0 — Environment validation and random agent

**Objective:** Verify installation, simulation stepping, observations/actions, and visualization before trying to train a neural network.

**Implementation:** `lunar_lander/random_agent.py` creates `LunarLander-v3`, repeatedly samples random discrete actions, prints the initial observation, and accumulates the total episode reward.

**Observed episodes:**

| Episode | Length (steps) | Total reward |
|---|---:|---:|
| 1 | 119 | -132.05 |
| 2 | 101 | -310.66 |
| 3 | 98 | -282.35 |

**Interpretation:** The simulation successfully responds to actions; the random pilot performs poorly in these three observed attempts. Three episodes are **not** a statistically useful random-policy benchmark.

**Decision:** Keep this as an environment smoke test and proceed to train PPO. Resolve the missing graphics window before visual inspection; rendering was subsequently confirmed working.

### E1 — Original PPO baseline: initial ~300k steps

**Objective:** Test whether unmodified PPO can learn useful landing behavior without custom rewards.

**Implementation:** `train.py` trained PPO from scratch on `Monitor(gym.make("LunarLander-v3"))` with standard rewards and the shared hyperparameters. Requested `300_000` steps; PPO completed **301,056** because training proceeds in complete 2,048-step rollouts. Model checkpoint: `ppo_lunar_lander.zip`.

**Training snapshot at 225,280 steps:**

- `ep_rew_mean`: **49.2**
- `ep_len_mean`: **723**
- `explained_variance`: **0.827**
- `approx_kl`: **0.00608**

**Final training snapshot at 301,056 steps:**

- `ep_rew_mean`: **74.1**
- `ep_len_mean`: **662**
- `explained_variance`: **0.740**
- `approx_kl`: **0.00949**
- Elapsed training time: **420 seconds (~7 minutes)**

**First rendered evaluation (five episodes):** The lander generally stayed upright, approached the platform, and slowed its descent. Horizontal-speed control was not clearly established; at least one landing was visually observed. This was **qualitative**, not a reliable success-rate estimate.

**Quantitative evaluation (100 episodes, original environment):**

| Metric | Result |
|---|---:|
| Mean original reward | **81.05** |
| Reward standard deviation | **117.58** |
| Mean episode length | **612.6** |
| Strict landings | **16 / 100 (16%)** |
| Episodes scoring >= 200 | **15 / 100** |

**Interpretation:** PPO learned behavior that is visibly better than random actions, but the resulting policy is unreliable and scores vary widely.

**Decision:** Preserve the 300k checkpoint. Continue training **the same model** to approximately 1 million total steps before changing the reward function, isolating the effect of additional training experience.

### E2 — Original PPO baseline: continued to ~1M total steps

**Objective:** Determine whether additional experience improves the original PPO policy without altering the environment or rewards.

**Implementation:** `continue_training.py` loads the 300k checkpoint and calls `learn(total_timesteps=700_000, reset_num_timesteps=False, tb_log_name="baseline_ppo")`. Saved separately as `ppo_lunar_lander_1m.zip`.

**Training:** approximately **1 million cumulative steps**. A TensorBoard comparison showed final training `ep_rew_mean` around **174.16** near 1,001,472 steps. This is a **training** value, not the evaluation mean.

**Quantitative evaluation (100 episodes):**

| Metric | Result |
|---|---:|
| Mean original reward | **198.27** |
| Reward standard deviation | **76.49** |
| Mean episode length | **450.1** |
| Safe rests | **85 / 100** |
| Safe rests with center on pad | **61 / 100** |
| Safe rests with both legs contacting terrain | **73 / 100** |
| Strict landings | **54 / 100 (54%)** |
| Off-pad safe rests | **24 / 100** |
| On-pad safe rests, insufficient leg contact | **7 / 100** |
| No safe rest | **15 / 100** |
| Episodes scoring >= 200 | **77 / 100** |

**Change versus 300k:** Mean evaluation reward increased by **117.22** (81.05 -> 198.27), strict landing rate rose **38 percentage points** (16% -> 54%), and mean episode length decreased (612.6 -> 450.1).

**Unexpected result and diagnostic:** 77 episodes scored >= 200, but only 54 met the strict criterion. High-reward episodes were inspected with final-state debug logging. Examples:

| Episode | Original reward | Final center x | Pad x-range | Legs in contact | Physical interpretation |
|---|---:|---:|---|---|---|
| 74 | 248.33 | 7.74 | [8, 12] | [True, True] | Resting slightly left of pad |
| 80 | 216.60 | 10.91 | [8, 12] | [False, True] | Center on pad, one leg not contacting |
| 81 | 215.06 | 14.31 | [8, 12] | [True, True] | Resting well right of pad |
| 82 | 213.58 | 12.01 | [8, 12] | [True, True] | Resting just beyond right boundary |

For all four, `terminated=True`, `truncated=False`, `game_over=False`, and `lander.awake=False`. Thus **high reward does not imply our strict landing definition**. There were **24 safe off-pad rests**, representing 24% of all evaluation attempts (about 28% of safe rests).

**Interpretation:** The longer-trained agent is relatively effective at coming to rest but still has an accuracy problem. The built-in terminal handling/reward does not directly enforce the strict center-on-pad, two-leg-contact metric.

**Decision:** Keep the ~1M unmodified PPO checkpoint as the **reference baseline**. Refine evaluation into separate safe-rest, pad-position, and leg-contact metrics; test whether rewarding pad accuracy more explicitly can help. Avoid relying only on `ep_rew_mean` or the count of episodes scoring >= 200.

### E3 — Terminal precision reward (fresh PPO, ~1M steps)

**Hypothesis:** Explicitly rewarding safe rests on the pad and penalizing safe rests outside it will improve strict landing success.

**Implementation:** `PrecisionLandingReward(gym.Wrapper)` in `rewards.py` modifies the *original* reward **only at a safe-rest terminal step**:

```python
if safe_rest:
    if on_pad:
        reward += 50.0
    else:
        reward -= 50.0
```

Nominal terminal safe-rest reward adjustment: **+50 on pad** versus **-50 off pad**. The rest of Gymnasium's step reward is unchanged. This **changes the learning objective**.

**Code/debug note:** The wrapper initially failed during construction because `super.__init__(env)` was written instead of `super().__init__(env)`. Correcting the parentheses fixed the Python initialization error.

**Training:** `train_precision.py` initialized a **new** PPO model with the same configuration, wrapped environment as `Monitor(PrecisionLandingReward(gym.make(...)))`, seed 42, and a requested 1,000,000 steps. Saved as `ppo_precision_1m.zip`. TensorBoard training `ep_rew_mean` near 1M was about **57.04**, with substantial earlier fluctuation; this is a *shaped training reward* and is not directly comparable to the baseline's training reward.

**Evaluation:** Unwrapped original LunarLander environment; the identical 100 evaluation seeds and deterministic prediction.

| Metric | Result |
|---|---:|
| Mean **original** reward | **125.12** |
| Reward standard deviation | **110.77** |
| Mean episode length | **427.3** |
| Safe rests | **70 / 100** |
| Safe rests with center on pad | **22 / 100** |
| Safe rests with both legs contacting terrain | **56 / 100** |
| Strict landings | **16 / 100 (16%)** |
| Off-pad safe rests | **48 / 100** |
| On-pad safe rests, insufficient leg contact | **6 / 100** |
| No safe rest | **30 / 100** |
| Episodes scoring >= 200 | **32 / 100** |

**Interpretation:** This single trained model was worse than the original baseline: **16% versus 54%** strict success, **48 versus 24** off-pad safe rests, and fewer safe rests overall. The terminal bonus/penalty supplies a sparse signal, so assigning credit to actions much earlier in the flight may be difficult. Training stochasticity and other optimization effects are also plausible contributors; **the logs do not identify a unique cause**.

**Decision:** Do **not** overwrite the baseline or claim that all terminal shaping is ineffective. Preserve this run as a negative result. Next try a theoretically motivated *dense* shaping signal that rewards progress over transitions, while keeping the original task objective.

### E4 — Potential-based reward shaping (fresh PPO, ~1M steps)

**Hypothesis:** Denser feedback about horizontal position and velocity may improve PPO learning and perhaps landing accuracy without intentionally changing the underlying optimal policy.

**Implementation:** `PotentialBasedReward(gym.Wrapper)` in `rewards.py` uses the observation's normalized horizontal position `x` and horizontal velocity `vx`:

\[
\Phi(s) = -20|x| - 10|v_x|
\]

\[
F(s_t,s_{t+1}) = \gamma\Phi(s_{t+1}) - \Phi(s_t),\quad \gamma=0.99
\]

The environment reward becomes `original_reward + F`. The wrapper stores the previous potential after `reset()` and sets the next potential to **zero at termination or truncation**, per the implementation discussed.

**Training:** `train_potential.py` initializes a **new** PPO with the shared hyperparameters, `Monitor(PotentialBasedReward(gym.make("LunarLander-v3"), gamma=0.99))`, and `tb_log_name="potential_reward"`. Requested 1,000,000 steps; reported actual **1,001,472** steps. Saved as `ppo_potential_1m.zip`.

**Final training-log snapshot (1,001,472 steps):**

| Metric | Value |
|---|---:|
| `ep_rew_mean` | **182** (shaped training reward) |
| `ep_len_mean` | **496** |
| `approx_kl` | **0.01123** |
| `clip_fraction` | **0.0790** |
| `explained_variance` | **0.773** |
| `entropy_loss` | **-0.454** |
| Processing speed | **~455 steps/s** |
| Elapsed time | **2,199 s (~36m 39s)** |

The final eight logged mean training rewards fluctuated roughly from **182 to 193**; this does not prove convergence. These training metrics cannot by themselves establish landing precision.

**Evaluation (unchanged original environment, 100 seeds):**

| Metric | Result |
|---|---:|
| Mean **original** reward | **157.57** |
| Reward standard deviation | **87.24** |
| Mean episode length | **482.3** |
| Safe rests | **77 / 100** |
| Safe rests with center on pad | **23 / 100** |
| Safe rests with both legs contacting terrain | **59 / 100** |
| Strict landings | **21 / 100 (21%)** |
| Off-pad safe rests | **54 / 100** |
| On-pad safe rests, insufficient leg contact | **2 / 100** |
| No safe rest | **23 / 100** |
| Episodes scoring >= 200 | **41 / 100** |

**Interpretation:** Potential-based shaping improved the strict landing rate relative to the terminal-shaping run (**21% versus 16%**) but was still substantially below the unmodified PPO baseline (**54%**). Most safe rests in this run were outside the pad (54 of 77).

**Important theoretical caveat:** Under appropriate discounting, proper treatment of terminal states, and other potential-based shaping assumptions, the discounted shaping terms telescope:

\[
\sum_{t=0}^{T-1}\gamma^t F_t
= -\Phi(s_0) + \gamma^T\Phi(s_T)
\]

With a zero terminal potential, the added discounted return is `-Phi(s_0)` for a given initial state. Thus, this shaping method is intended to change *learning feedback* without making pad accuracy a different optimal objective. **It cannot be assumed to fix misalignment between Gymnasium's original objective and our stricter success definition.** The treatment of time-limit truncation in the implementation deserves additional scrutiny before making a formal policy-invariance claim.

**Decision:** Preserve the checkpoint and results. **Do not immediately start another million-step run.** First diagnose the *trajectory and spatial distribution* of off-pad resting episodes, then define and test an objective explicitly aligned with precision landing. Later repeat training with multiple seeds to distinguish robust effects from run-specific variation.

---

## 6. Consolidated results

All evaluation values below use **100 episodes**, the **same evaluation seeds (1000–1099)**, **deterministic policy actions**, and the **original, unmodified LunarLander-v3 environment and rewards**.

| Metric | PPO ~300k | PPO ~1M (baseline) | Terminal precision ~1M | Potential shaping ~1M |
|---|---:|---:|---:|---:|
| Mean original reward | 81.05 | **198.27** | 125.12 | 157.57 |
| Reward SD | 117.58 | **76.49** | 110.77 | 87.24 |
| Mean episode length | 612.6 | 450.1 | **427.3** | 482.3 |
| Safe rests | Not recorded | **85** | 70 | 77 |
| Safe rests with center on pad | Not recorded | **61** | 22 | 23 |
| Safe rests with both legs in contact | Not recorded | **73** | 56 | 59 |
| Strict landings | 16 | **54** | 16 | 21 |
| Off-pad safe rests | Not recorded | **24** | 48 | 54 |
| Safe on-pad rests with bad leg contact | Not recorded | 7 | 6 | **2** |
| No safe rest | Not recorded | **15** | 30 | 23 |
| Episodes scoring >= 200 | 15 | **77** | 32 | 41 |

**Reading the table:** Bold in metrics where **higher** is generally desirable indicates the best observed model; for failure counts and variability **lower** is preferable. Mean episode length has **no universal best direction**, since faster completion could be safe landing or early failure.

### Three main observations

1. **More baseline training helped in this run.** Strict landing success improved 16% -> 54%, and original mean return increased 81.05 -> 198.27.
2. **High reward and strict physical success are not equivalent.** The best model produced 77 episodes scoring >= 200 but only 54 strict landings; recorded high-scoring off-pad safe rests demonstrate the gap.
3. **Neither of the two reward modifications improved strict success in its single reported training run.** Terminal shaping: 16%; potential-based shaping: 21%; standard ~1M PPO: 54%.

### Safe-rest conditional precision

These are derived ratios, **not additional experiments**:

| Policy | Strict landings / safe rests | Conditional strict-landing rate |
|---|---|---:|
| Standard PPO ~1M | 54 / 85 | **63.5%** |
| Terminal precision reward | 16 / 70 | **22.9%** |
| Potential-based shaping | 21 / 77 | **27.3%** |

This supports prioritizing landing accuracy, while acknowledging that unsafe terminations also remain a meaningful issue.

---

## 7. Decision log

| Decision | Rationale | Status |
|---|---|---|
| **DEC-001 — Start in 2D Gymnasium instead of 3D Godot** | Learn RL, evaluation, and rewards before adding engine- and simulation-integration complexity | Adopted |
| **DEC-002 — Validate the environment with random actions** | Separate installation/physics/rendering issues from learning issues | Completed |
| **DEC-003 — Begin with unmodified PPO and original rewards** | Establish a reference without objective-engineering confounds | Completed |
| **DEC-004 — Continue 300k checkpoint to ~1M** | Test the effect of additional training while keeping reward and architecture fixed | Completed; improvement observed |
| **DEC-005 — Evaluate 100 seeded episodes on original rewards** | Obtain comparable numerical performance rather than judging videos or shaped training scores | Adopted for all experiments |
| **DEC-006 — Define a stricter physical success metric** | Separate high reward / safe rest from center-on-pad, two-leg-contact landing | Adopted; metric caveats documented |
| **DEC-007 — Test terminal precision bonus/penalty** | Directly emphasize safe on-pad versus safe off-pad terminal outcome | Completed; this run underperformed |
| **DEC-008 — Test potential-based shaping** | Examine whether dense progress feedback improves learning | Completed; better than terminal experiment but worse than baseline |
| **DEC-009 — Stop changing rewards until behavior is diagnosed** | Prevent arbitrary hyperparameter/reward searches; investigate off-pad outcomes and measurement first | Current decision |
| **DEC-010 — Repeat experiments across training seeds before general claims** | One training seed per configuration is not sufficient to establish a method's general superiority | Planned |

---

## 8. What we learned technically

**Reinforcement learning fundamentals:** observation, action, reward, episode, policy, inference (`predict`) versus optimization (`learn`), and PPO rollout batches.

**Model selection and monitoring:** training reward versus evaluation return; `ep_len_mean`, `approx_kl`, `clip_fraction`, entropy, explained variance; saving independent checkpoints.

**Experimental design:** change one main variable at a time; compare policies on the same unmodified test environment and starting seeds; distinguish a physical success metric from reward score.

**Reward engineering:** terminal bonuses are sparse and may be difficult to attribute to early actions; potential-based shaping changes learning feedback but does not automatically repair a poorly aligned target objective.

**Debugging:** interpreted a Python traceback to fix a missing `super()` call; investigated high-scoring off-pad landings by inspecting simulation internals.

**Scientific caution:** Current comparisons show the behavior of the *particular trained policies* reported here. They do not establish statistically robust conclusions over different model initializations, training seeds, or future code/library versions.

---

## 9. Next steps (planned, not completed)

### Priority 1 — Diagnose misses before another training run

For every safe off-pad rest, record the final center displacement relative to the helipad center:

```python
pad_center = (sim.helipad_x1 + sim.helipad_x2) / 2
horizontal_error = sim.lander.position.x - pad_center
```

Then summarize **left-side versus right-side misses**, displacement magnitudes, and proximity to the pad boundary. Extend later to trajectories (position, horizontal/vertical velocity, angle, chosen action across time) for selected episodes to distinguish overshoot from early stopping and poor lateral control.

### Priority 2 — Strengthen evaluation tooling

- Save a row per episode to CSV (policy, training seed, evaluation seed, original return, length, outcome category, final position/error).
- Keep a fixed baseline test set for comparability and consider a separate held-out evaluation seed set for final reporting.
- Log separate counts of crashes, out-of-bounds, and time-limit truncations within the current broad **no safe rest** category.
- Consider a precise pad-foot-contact geometry metric if the strict definition becomes central to the final report.
- Add tests for outcome classification and reward-wrapper behavior, including terminal versus truncated transitions.

### Priority 3 — Design a better-aligned precision objective

Use the diagnosed errors to design and test an explicit objective for accurate touchdown, rather than assuming that the original terminal bonus or potential-based progress signal fully captures the goal. Test reward calculations on controlled synthetic transitions before launching full training.

### Priority 4 — Improve experimental validity

Run multiple independent **training** seeds per method, with matched budgets and evaluation seeds; report means, uncertainty, and distribution of strict landing success. Compare policies on the same original reward and physical metrics.

### Priority 5 — Prepare portfolio materials and future 3D extension

Prepare a clear README, short landing/failure video clips, training curves, reproducible commands, experiment tables, and selected trajectories. Consider a later Godot 4 + Godot RL Agents environment with continuous thrust/gimbal control, fuel limitations, and robustness perturbations **after** a solid 2D experimental write-up.

---

## 10. Reproducibility commands

Run from the repository root (`C:\ResumeProjects\RocketRL`) with the project virtual environment installed.

```powershell
# Verify the environment / inspect the random agent
.\.venv\Scripts\python.exe lunar_lander\random_agent.py

# Original PPO: fresh ~300k run
.\.venv\Scripts\python.exe lunar_lander\train.py

# Continue original PPO to ~1M total
.\.venv\Scripts\python.exe lunar_lander\continue_training.py

# Precision terminal-reward experiment (fresh policy)
.\.venv\Scripts\python.exe lunar_lander\train_precision.py

# Potential-based reward experiment (fresh policy)
.\.venv\Scripts\python.exe lunar_lander\train_potential.py

# Quantitative evaluation (set the desired model path inside the script)
.\.venv\Scripts\python.exe lunar_lander\evaluate_metrics.py

# Compare training logs
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs
```

**Important:** `evaluate_metrics.py` currently loads a checkpoint from a hardcoded model path. Set it to the correct model before each run, and leave evaluation `env = gym.make("LunarLander-v3")` **without a custom reward wrapper**. The evaluation already uses `seed=1000+episode` and deterministic actions.

---

## 11. Interpretation and current conclusion

The best observed policy is **standard PPO trained for approximately one million steps**, with **54/100 strict landings**, **85/100 safe rests**, and **198.27 mean original episode reward** on the fixed 100-episode evaluation set. Continuing the original policy from 300k to ~1M steps improved measured performance substantially.

The terminal precision-reward experiment and potential-based shaping experiment **did not improve strict landing success** in their respective single-seed runs. Their evaluations also illustrate why training reward, original-environment reward, and strict physical landing criteria must be kept separate.

The next justified step is **diagnostic analysis**, not another uncontrolled million-step reward experiment. The central challenge remains the mismatch between **safe resting** and **precise resting on the target platform**. The project currently provides a reproducible initial study and a strong foundation for subsequent custom-environment work, but stronger scientific claims require a more precise success detector and multiple training seeds.
