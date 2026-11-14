if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


from .environment import make_environment

def main():
    env = make_environment(render_mode="human")
    env.action_space.seed(42)

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
