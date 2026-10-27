import gymnasium as gym
from statistics import mean, stdev
from stable_baselines3 import PPO


def main():
    model = PPO.load(
        "models/ppo_potential_1m",
        device="cpu"
    )

    env = gym.make("LunarLander-v3")

    rewards = []
    episode_lengths= []
    
    successful_landings = 0
    safe_rest_landings = 0
    safe_rest_on_pad = 0
    safe_rest_both_legs = 0

    safe_rest_off_pad = 0
    safe_rest_bad_contact = 0
    unsafe_terminations = 0


    benchmark_scores = 0

    num_episodes = 100

    
    left_misses = []
    right_misses = []
    pad_miss_distances = []
    center_errors = []


    for episode in range(num_episodes):
        obs, info = env.reset(seed = 1000 + episode)

        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(
                obs,
                deterministic = True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward >= 200:
            benchmark_scores += 1

        #Check the final physical state.
        sim = env.unwrapped
        
        # 1. Did the lander come to rest without crashing?
        safe_rest = (
            terminated
            and not truncated
            and not sim.game_over
            and not sim.lander.awake
        )

        # 2. Is the lander's center above the platform?
        on_pad = (
            sim.helipad_x1
            <= sim.lander.position.x
            <= sim.helipad_x2
        )

        
        
        if safe_rest and not on_pad:
            x = sim.lander.position.x

            pad_left = sim.helipad_x1
            pad_right = sim.helipad_x2
            pad_center = (pad_left + pad_right) / 2

            center_error = x - pad_center

            if x < pad_left:
                miss_distance = pad_left - x
                left_misses.append(miss_distance)
            else:
                miss_distance = x - pad_right
                right_misses.append(miss_distance)

            pad_miss_distances.append(miss_distance)
            center_errors.append(center_error)



        # 3. Are both landing legs touching the ground?
        both_legs = all(
            leg.ground_contact for leg in sim.legs
        )

        # 4. Does it satisfy our stricter landing definition?
        landed = safe_rest and on_pad and both_legs

        
        if safe_rest:
            safe_rest_landings += 1

            if on_pad:
                safe_rest_on_pad += 1

            if both_legs:
                safe_rest_both_legs += 1

            # Categorize the outcome
            if not on_pad:
                safe_rest_off_pad += 1

            elif not both_legs:
                safe_rest_bad_contact += 1

            else:
                successful_landings += 1

        else:
            unsafe_terminations += 1



        # Inspect episodes that score highly but fail our landing check
        if total_reward >= 200 and not landed:
            print(f"\nEpisode {episode + 1}: HIGH REWARD, NOT LANDED")
            print(f"Reward: {total_reward:.2f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Game over: {sim.game_over}")
            print(f"Lander awake: {sim.lander.awake}")
            print(f"Leg contacts: {[leg.ground_contact for leg in sim.legs]}")
            print(f"Horizontal position: {sim.lander.position.x:.2f}")
            print(f"Helipad bounds: {sim.helipad_x1:.2f}, {sim.helipad_x2:.2f}")

    print("\n===== ROCKETRL EVALUATION =====")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean(rewards):.2f}")
    print(f"Reward standard deviation: {stdev(rewards):.2f}")
    print(f"Average episode length: {mean(episode_lengths):.1f}")
    print(f"Episodes scoring 200+: {benchmark_scores}/{num_episodes}")
        
    print("\n===== LANDING ANALYSIS =====")

    print(f"Safe rests: {safe_rest_landings}/{num_episodes}")
    print(f"Safe rests on pad: {safe_rest_on_pad}/{num_episodes}")
    print(f"Safe rests with both legs: {safe_rest_both_legs}/{num_episodes}")

    print("\n===== EXCLUSIVE OUTCOMES =====")

    print(f"Strict successful landings: {successful_landings}")
    print(f"Safe rests outside pad: {safe_rest_off_pad}")
    print(f"Safe rests on pad with insufficient leg contact: {safe_rest_bad_contact}")
    print(f"No safe rest: {unsafe_terminations}")


    
    print("\n===== HORIZONTAL ERROR ANALYSIS =====")

    print(f"Left misses: {len(left_misses)}")
    print(f"Right misses: {len(right_misses)}")

    if pad_miss_distances:
        print(
            f"Average distance outside pad: "
            f"{mean(pad_miss_distances):.2f}"
        )

        print(
            f"Maximum distance outside pad: "
            f"{max(pad_miss_distances):.2f}"
        )

        near_misses = sum(
            d <= 0.5 for d in pad_miss_distances
        )

        print(f"Near misses (within 0.5 units): {near_misses}")

    assert (
        len(left_misses) + len(right_misses)
        == safe_rest_off_pad
    )


    assert (
        successful_landings
        + safe_rest_off_pad
        + safe_rest_bad_contact
        + unsafe_terminations
        == num_episodes
    )

    assert safe_rest_landings == (
        successful_landings
        + safe_rest_off_pad
        + safe_rest_bad_contact
    )

    assert safe_rest_on_pad == (
        successful_landings + safe_rest_bad_contact
    )

    env.close()


if __name__ == "__main__":
    main()
