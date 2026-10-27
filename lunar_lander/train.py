from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def main():
    #Create directories for models and training logs
    Path("models").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    #Create environment without renering for faster training
    env = Monitor(gym.make("LunarLander-v3"))

    #Initial PPO with a multilayer perceptron policy
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

    #Train for 300000 simulation steps
    model.learn(total_timesteps = 300_000, tb_log_name="baseline_ppo",)

    #Save trained neural network
    model.save("models/ppo_lunar_lander")

    env.close()
    print("Training completed. Model saved.")


if __name__ == "__main__":
    main()