from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rewards import PrecisionLandingReward

def main():
    Path("models").mkdir(exist_ok=True)

    # Original environment with our custom reward wrapper
    env = Monitor(PrecisionLandingReward(
        gym.make("LunarLander-v3")
    ))

    # New model, same PPO configuration as our baseline
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

    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="precision_reward",
    )

    model.save("models/ppo_precision_1m")

    env.close()

if __name__ == "__main__":
    main()