import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    #Create the original training environment
    env = Monitor(gym.make("LunarLander-v3"))

    #Load my previous trained AI
    model = PPO.load(
        "models/ppo_lunar_lander",
        env=env,
        device="cpu",
        tensorboard_log="runs/"
    )

    model.learn(
        total_timesteps=700_000,
        reset_num_timesteps=False,
        tb_log_name="baseline_ppo"
    )

    #Preserve the new model separately
    model.save("models/ppo_lunar_lander_1m")

    env.close()

    print("Extended training completed.")


if __name__ == "__main__":
    main()