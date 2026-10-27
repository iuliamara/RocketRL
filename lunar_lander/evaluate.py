import gymnasium as gym
from stable_baselines3 import PPO
from statistics import mean


def main():
    #Load the neural network we trained
    model = PPO.load("models/ppo_lunar_lander", device="cpu")

    #Create the environment with visual rendering
    env = gym.make("LunarLander-v3", render_mode="human")

    episode_rewards = []

    #Let our AI attempt five landings
    for episode in range(5):
        observation, info = env.reset(seed=100 + episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            #Let the trained neural network choose an action
            action, _ = model.predict(observation,deterministic=True)

            #Execute the action in the simulator
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        print(f"Episode {episode + 1}")
        print(f"Total reward: {total_reward: .2f}")
        print(f"Steps: {steps}")
        print("--------------------------------")

    print(f"Mean reward: {mean(episode_rewards): .2f}")
    env.close()


if __name__ == "__main__":
    main()
