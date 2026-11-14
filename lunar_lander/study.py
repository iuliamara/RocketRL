import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter

from .config import HELD_OUT_SEEDS, HISTORICAL_SEEDS, TrainingConfig
from .evaluate_metrics import evaluate_to_directory
from .io import sha256, write_csv, write_json
from .train import REWARDS, train


DEFAULT_REWARDS = ("baseline", "potential", "landing-objective")
DEFAULT_SEEDS = (42, 43, 44)
METRICS = (
    "mean_original_return", "strict_landing_rate", "safe_rest_rate", "safe_rest_position_mae",
    "mean_steps", "mean_engine_cost",
)


def stats(values):
    values = [value for value in values if value is not None]
    result = {
        "training_seeds": len(values), "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None, "ci95_low": None, "ci95_high": None,
    }
    if len(values) >= 3:
        rng = Random(313)
        replicates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10_000))
        result.update(ci95_low=replicates[250], ci95_high=replicates[9749])
    return result


def summarize_study(runs, rewards, splits):
    aggregates, contrasts = [], []
    for split in splits:
        by_recipe = {
            reward: {row["training_seed"]: row for row in runs if row["split"] == split and row["recipe"] == reward}
            for reward in rewards
        }
        for reward, by_seed in by_recipe.items():
            for metric in METRICS:
                aggregates.append({
                    "recipe": reward, "split": split, "metric": metric,
                    **stats([row[metric] for row in by_seed.values()]),
                })
            if reward == "baseline" or "baseline" not in by_recipe:
                continue
            baseline = by_recipe["baseline"]
            for metric in METRICS:
                deltas = [
                    row[metric] - baseline[seed][metric] for seed, row in by_seed.items()
                    if seed in baseline and row[metric] is not None and baseline[seed][metric] is not None
                ]
                contrasts.append({
                    "recipe": reward, "reference": "baseline", "split": split, "metric": metric,
                    **stats(deltas),
                })
    return aggregates, contrasts


def run_study(output, recipes, seeds, steps, episodes):
    splits = {"historical": list(HISTORICAL_SEEDS)[:episodes], "held_out": list(HELD_OUT_SEEDS)[:episodes]}
    manifest = {"recipes": recipes, "training_seeds": seeds, "steps": steps, "splits": splits}
    output.mkdir(exist_ok=False, parents=True)
    write_json(output / "study.json", manifest)
    runs = []
    for recipe in recipes:
        for seed in seeds:
            config = TrainingConfig(reward=recipe, seed=seed, steps=steps)
            run = output / config.run_id
            start = perf_counter()
            checkpoint = train(config, run / "training")
            train_seconds = perf_counter() - start
            actual_steps = json.loads((checkpoint.parent / "complete.json").read_text())["actual_steps"]
            for split, test_seeds in splits.items():
                summary = evaluate_to_directory(checkpoint, test_seeds, run / split)
                runs.append({
                    "recipe": recipe, "training_seed": seed, "split": split, "actual_steps": actual_steps,
                    "checkpoint_sha256": sha256(checkpoint), "train_seconds": train_seconds,
                    **{key: summary[key] for key in METRICS},
                })
    aggregates, contrasts = summarize_study(runs, recipes, splits)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "aggregates.csv", aggregates)
    if contrasts:
        write_csv(output / "paired_differences.csv", contrasts)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a matched multi-seed PPO study.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipes", choices=REWARDS, nargs="+", default=DEFAULT_REWARDS)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per evaluation split and checkpoint.")
    parser.add_argument("--plan", action="store_true", help="Print the run count and budget without training.")
    args = parser.parse_args()
    if len(set(args.recipes)) != len(args.recipes) or len(set(args.train_seeds)) != len(args.train_seeds):
        parser.error("Recipes and training seeds must be unique.")
    if args.steps < 1 or min(args.train_seeds) < 0 or not 1 <= args.episodes <= 100:
        parser.error("Train seeds must be nonnegative, steps positive, and episodes between 1 and 100.")
    count = len(args.recipes) * len(args.train_seeds)
    rollout = TrainingConfig().n_steps
    rounded_steps = ((args.steps + rollout - 1) // rollout) * rollout
    print(f"{count} runs x {rounded_steps:,} training steps and {2 * count * args.episodes:,} evaluation episodes.", flush=True)
    if not args.plan:
        run_study(args.output, args.recipes, args.train_seeds, args.steps, args.episodes)


if __name__ == "__main__":
    main()
