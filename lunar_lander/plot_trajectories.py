
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("results/potential_trajectories.csv")
OUTPUT_DIR = Path("results/trajectory_plots")

EPISODES = [2, 34, 55, 11]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read trajectory measurements from CSV
    data = pd.read_csv(DATA_PATH)

    # GRAPH 1: Horizontal position versus vertical position
    plt.figure(figsize=(10, 7))

    # Highlight the landing pad's horizontal interval
    plt.axvspan(
        8, 12,
        alpha=0.15,
        label="Target horizontal range"
    )

    for episode in EPISODES:
        trajectory = data[data["episode"] == episode]

        outcome = trajectory["outcome"].iloc[0]

        plt.plot(
            trajectory["x"],
            trajectory["y"],
            label=f"Episode {episode} ({outcome})"
        )

    plt.xlabel("Horizontal position (Box2D units)")
    plt.ylabel("Vertical position (Box2D units)")
    plt.title("Rocket Landing Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "position_trajectories.png")
    plt.close()

    # GRAPH 2: Horizontal velocity over time
    plt.figure(figsize=(10, 6))

    for episode in EPISODES:
        trajectory = data[data["episode"] == episode]

        plt.plot(
            trajectory["step"],
            trajectory["vx"],
            label=f"Episode {episode}"
        )

    plt.axhline(0, linestyle="--", alpha=0.6)

    plt.xlabel("Simulation step")
    plt.ylabel("Horizontal velocity (Box2D units/s)")
    plt.title("Horizontal Velocity During Landing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "horizontal_velocity.png")
    plt.close()

    print("Trajectory plots generated successfully!")


if __name__ == "__main__":
    main()
