import gymnasium as gym

from .metrics import landing_state


class PrecisionLandingReward(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not truncated:
            state = landing_state(self.unwrapped, terminated, truncated)
            if state.safe_rest:
                reward += 50 if state.on_pad else -50
        return obs, reward, terminated, truncated, info


class PotentialBasedReward(gym.Wrapper):
    def __init__(self, env, gamma=0.99, *, legacy_zero_on_truncation=False):
        super().__init__(env)
        self.gamma = gamma
        self.legacy_zero_on_truncation = legacy_zero_on_truncation
        self.previous_potential = None

    def potential(self, observation):
        return -20.0 * abs(float(observation[0])) - 10.0 * abs(float(observation[2]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_potential = self.potential(obs)
        if terminated or (truncated and self.legacy_zero_on_truncation):
            next_potential = 0.0
        reward += self.gamma * next_potential - self.previous_potential
        self.previous_potential = next_potential
        return obs, reward, terminated, truncated, info


class LandingObjective(gym.Wrapper):
    """An explicit precision, stability, and engine-use objective."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sim = self.unwrapped
        state = landing_state(sim, terminated, truncated)
        # Gymnasium's normalized horizontal speed and height proxy.
        x, height, vx = map(float, obs[:3])
        if height <= 0.6 and obs[3] < 0 and x * vx > 0:
            reward -= 0.10 * min(abs(vx), 1.0)
        # Original LunarLander already penalizes engine firing. This doubles that cost.
        reward -= 0.3 if action == 2 else 0.03 if action in (1, 3) else 0.0
        reward -= 0.01
        if terminated or truncated:
            reward += 50.0 if state.outcome == "strict_landing" else -50.0
        return obs, reward, terminated, truncated, info
