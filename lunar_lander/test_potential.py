
import gymnasium as gym
from rewards import PotentialBasedReward

env = PotentialBasedReward(
    gym.make("LunarLander-v3"),
    gamma=0.99
)

obs, info = env.reset(seed=42)

for step in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step + 1}: Reward = {reward:.3f}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
