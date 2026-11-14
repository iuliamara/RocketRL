# Technical explanations

## States, actions, and returns

A Markov decision process (MDP) specifies a state, available actions, transition dynamics, and reward. In an MDP the current state and action determine the distribution of the next state and reward. A simulator may contain information missing from the agent's observation, such as terrain geometry beyond what the 2D lander observes. An MLP policy treats each observation as sufficient for choosing the next action. A history-based policy or additional sensors are alternatives when that approximation is weak.

The discounted return from step $t$ is $G_t=\sum_{k\ge0}\gamma^k r_{t+k}$. The policy $\pi_\theta(a|s)$ chooses an action distribution. A value function $V^\pi(s)$ estimates future return from a state under that policy. The advantage $A^\pi(s,a)$ measures how much an action improves on that state value in expectation. LunarLander actions are categorical. The Godot policy produces three continuous commands and PPO clips commands at the action space bounds.

## PPO and credit assignment

PPO collects rollouts using the current policy, then performs several minibatch optimization passes. For probability ratio $\rho_t=\pi_\theta(a_t|s_t)/\pi_{\rm old}(a_t|s_t)$, PPO maximizes the average of

$$\min(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t).$$

The clipped term limits the incentive for large changes on sampled actions. The value network minimizes a regression loss against estimated returns. A simple MLP is sufficient to represent a policy over these small observation vectors. PPO provides a common algorithm across both action spaces. SAC would be an off-policy alternative for the continuous task with different data reuse and tuning.

Generalized advantage estimation uses one-step residuals $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ and sums future residuals with weights $(\gamma\lambda)^k$. A smaller $\lambda$ puts more weight on the learned value estimate. It reduces variance at the cost of relying on a potentially biased critic. This helps credit assignment because the policy needs to connect a landing outcome to commands issued many steps earlier. The last term omits bootstrap value at true termination. A time-limit cutoff can retain bootstrap value at its final observation. [PPO paper](https://arxiv.org/abs/1707.06347) · [GAE paper](https://arxiv.org/abs/1506.02438) · [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

## Reward shaping and objective design

The historical potential is $\Phi(s)=-20|x|-10|v_x|$ using normalized observations. Adding $F_t=\gamma\Phi(s_{t+1})-\Phi(s_t)$ yields a discounted sum of shaping terms $-\Phi(s_0)+\gamma^T\Phi(s_T)$. A true terminal state has zero potential here. At a time-limit cutoff, keep $\Phi(s_T)$ because the shaped value $V^\pi(s_T)-\Phi(s_T)$ is bootstrapped. Its potential term cancels the remaining discounted boundary term. The shaping discount is the PPO discount so these cancellations use identical factors. If both flags are set, the true terminal behavior takes precedence. The legacy zero-on-cutoff option reproduces the historical recipe. [Potential-based shaping paper](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)

Policy invariance is a statement about the underlying discounted objective under the theorem's assumptions. Approximation, sampling, and finite training can produce different learned policies. If the primary objective needs to prefer strict landings, an explicit reward change expresses that preference directly. The historical precision wrapper awards +50 for safe on-pad rest and −50 for safe off-pad rest. The newer objective awards +50 for strict success and −50 on other episode endings, including runtime cutoff. At each step it charges 0.01 for elapsed time and adds the original action charge again. It charges up to 0.10 on descending transitions within the normalized height gate when horizontal velocity points away from pad center. The low-altitude threshold and weights are hypotheses for a follow-up study.

Reward hacking occurs when policy behavior exploits an incentive that has weak alignment with task intent. For example, an agent might rest early off-pad to avoid engine and time costs or trade excessive fuel for terminal bonuses. Compare original return, strict and off-pad outcomes, episode length, and engine use. A successful low-budget smoke test establishes execution. Learning a robust behavior requires longer runs and repeated seeds.

## Evaluating generalization

A test seed controls simulator initial conditions. A training seed affects model initialization, sampling, and training interactions. Repeating evaluation seeds controls the comparison between checkpoints. Repeating training seeds estimates variation across learned policies. The study computes per-seed summaries and resamples those summaries for bootstrap intervals. With three seeds the intervals have very low resolution. A second fixed test split measures new starts from the same family. Godot perturbation tests change one factor at a time, which helps isolate sensitivity. Joint perturbations and real hardware introduce different shifts.

## Rigid-body dynamics, thrust, and feedback

Godot integrates a body with linear position and velocity, orientation, and angular velocity. Translational acceleration depends on net force and mass, $a=F/m$. A torque applied to a rigid body changes angular velocity according to its inertia and orientation. Gravity acts along world −Y in RocketRL, with X and Z spanning the ground. Fuel changes the mass each tick. Throttle controls burn and available thrust. The two gimbal values tilt a local thrust vector, and the body orientation maps the vector to world coordinates. In this simplified model thrust acts through the center and a separate attitude PD controller applies bounded torque. A physical engine model would apply thrust at a nozzle offset and account for the coupled torque, changing center of mass, and aerodynamics.

The classical baseline derives a target vertical speed from height, compensates for gravity, and asks for horizontal acceleration proportional to displacement and velocity. The proportional term reduces position error. The derivative term damps drift and overshoot. The PD calculation uses simulator state and approximate force compensation. An integral term could remove steady-state bias but would add windup behavior when an actuator saturates. The benchmark uses simulated mass, engine efficiency, and wind in the observation, giving both policies access to those quantities.

## Simulation and policy integration

Godot runs its physics at 60 Hz. The policy selects a command every four ticks (15 Hz) until first contact, leaving bounds, or an episode limit. The returned terminal observation belongs to the ending episode. The adapter maps the protocol's `done` to Gymnasium's termination or truncation using the reason field, so SB3 can bootstrap limits correctly. Deterministic runs use the same reset seeds and fixed action durations. The headless path skips rendering and uses fixed engine timing. The visible path draws the same physics state.

A local framed JSON connection is easy to inspect and integrates the official Godot RL Agents Python protocol core without an export template or its generic vector wrapper. The tradeoffs are per-step serialization and a single engine process per environment. Parallel workers or a packed binary protocol could improve throughput. State normalization uses fixed scales and clipping. The observing policy uses the body's up direction, which suffices for the near-upright task but loses yaw. The simulator uses first contact for landing classification and a pre-impact velocity sample. It does not model flexible landing legs, sustained rest, sensor latency, or real rocket hardware. [Godot rigid-body documentation](https://docs.godotengine.org/en/4.5/classes/class_rigidbody3d.html) · [Godot RL Agents Python project](https://github.com/edbeeching/godot_rl_agents)
