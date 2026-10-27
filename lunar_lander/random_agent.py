import gymnasium as gym

def main():
    env = gym.make("LunarLander-v3", render_mode = "human")

    for episode in range(3):
        observation, info = env.reset(seed=42 + episode)
        total_reward=0
        steps = 0

        print(f"\nEpisode {episode + 1}")
        print(f"\nObservation {observation}")

        while True:
            #Choosse a random action: 0,1,2 or 3
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if(terminated or truncated):
                break

        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward: .2f}")

    env.close()


if __name__ == "__main__":
    main()