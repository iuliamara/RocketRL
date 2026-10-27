import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


MODEL_PATH = "models/ppo_lunar_lander_1m"
OUTPUT_PATH = Path("results/baseline_trajectories.csv")

NUM_EPISODES = 100

def main():
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)

    model = PPO.load(MODEL_PATH, device="cpu")
    env= gym.make("LunarLander-v3")
    columns= ["episode", "step", "x", "y", "vx", "vy", "angle", "action", "outcome"]

    try:
        with OUTPUT_PATH.open("w",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()

            for episode in range(NUM_EPISODES):
                obs,info = env.reset(seed=1000 +episode)

                trajectory = []
                terminated= False
                truncated= False

                while not (terminated or truncated):
                    action, _ = model.predict(obs,deterministic=True)
                    obs,reward,terminated,truncated,info =(
                        env.step(action)
                    )
                    body = env.unwrapped.lander

                    trajectory.append({
                        "episode": episode + 1,
                        "step": len(trajectory) + 1,
                        "x": float(body.position.x),
                        "y": float(body.position.y),
                        "vx": float(body.linearVelocity.x),
                        "vy": float(body.linearVelocity.y),
                        "angle": float(body.angle),
                        "action": int(action),
                    })

                sim = env.unwrapped

                safe_rest = (
                    terminated
                    and not truncated
                    and not sim.game_over
                    and not sim.lander.awake
                )

                on_pad = (
                    sim.helipad_x1
                    <= sim.lander.position.x
                    <= sim.helipad_x2
                )

                both_legs = all(
                    leg.ground_contact for leg in sim.legs
                )

                if not safe_rest:
                    outcome = "no_safe_rest"
                elif not on_pad:
                    outcome = "off_pad"
                elif not both_legs:
                    outcome = "bad_contact"
                else:
                    outcome = "strict_landing"

                for row in trajectory:
                    row["outcome"] = outcome
                    writer.writerow(row)

                print(
                    f"Episode {episode + 1}: "
                    f"{outcome}, {len(trajectory)} steps"
                )

    finally:
        env.close()

    print(f"\nTrajectories saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
