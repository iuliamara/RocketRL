if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


import argparse
import json
from pathlib import Path

from .config import ARTIFACT_DIR, HISTORICAL_SEEDS, MODEL_DIR
from .io import lunar_versions, sha256, write_csv, write_json
from .metrics import summarize
from .rollout import evaluate_checkpoint


def evaluate_to_directory(checkpoint, seeds, output, *, telemetry=False):
    seeds = list(seeds)
    model_digest = sha256(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    training_seed, episodes, frames = evaluate_checkpoint(checkpoint, seeds, telemetry=telemetry)
    write_csv(output / "episodes.csv", episodes)
    if frames:
        write_csv(output / "trajectory.csv", frames)
    summary = {
        "checkpoint": str(checkpoint),
        "sha256": model_digest,
        "training_seed": training_seed,
        "evaluation_seeds": seeds,
        "versions": lunar_versions(),
        **summarize(episodes),
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None, *, telemetry=False, default_model="ppo_potential_1m.zip"):
    parser = argparse.ArgumentParser(description="Evaluate PPO with original LunarLander rewards.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=MODEL_DIR / default_model)
    parser.add_argument("--seed-start", type=int, default=HISTORICAL_SEEDS.start)
    parser.add_argument("--episodes", type=int, default=len(HISTORICAL_SEEDS))
    parser.add_argument("--telemetry", action="store_true", default=telemetry)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed_start < 0:
        parser.error("Episodes must be positive and the starting seed nonnegative.")
    if not args.checkpoint.suffix:
        args.checkpoint = args.checkpoint.with_suffix(".zip")
    stop = args.seed_start + args.episodes
    default_name = f"{args.checkpoint.stem}-s{args.seed_start}-{stop - 1}"
    output = args.output or ARTIFACT_DIR / "eval" / default_name
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --output.")
    print(json.dumps(evaluate_to_directory(
        args.checkpoint, range(args.seed_start, stop), output, telemetry=args.telemetry
    ), indent=2))


if __name__ == "__main__":
    main()
