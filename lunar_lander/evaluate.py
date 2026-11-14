if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


import argparse
from pathlib import Path

from .config import MODEL_DIR
from .metrics import summarize
from .rollout import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Watch a checkpoint in the LunarLander simulator.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=MODEL_DIR / "ppo_lunar_lander.zip")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error("Episodes must be positive and the starting seed nonnegative.")
    _, episodes, _ = evaluate_checkpoint(
        args.checkpoint, range(args.seed_start, args.seed_start + args.episodes), render_mode="human"
    )
    for row in episodes:
        print(f"Seed {row['seed']}: {row['original_return']:.2f}, {row['steps']} steps, {row['outcome']}")
    print(f"Mean reward: {summarize(episodes)['mean_original_return']:.2f}")


if __name__ == "__main__":
    main()
