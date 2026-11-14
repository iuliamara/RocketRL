import argparse
from collections import Counter
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from statistics import mean
import subprocess
from time import perf_counter

import numpy as np

from lunar_lander.io import sha256, write_csv, write_json
from .controller import pd_action
from .environment import CONDITIONS, RocketLandingEnv


def simulator_versions(engine):
    output = subprocess.check_output([str(engine.resolve()), "--version"], text=True)
    return {
        "godot": output.strip(),
        "godot_rl": version("godot-rl"),
        "stable_baselines3": version("stable-baselines3"),
        "gymnasium": version("gymnasium"),
    }


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {
        "seed": args.seed_start, "requested_steps": args.steps, "n_steps": 1024,
        "condition": "nominal", **simulator_versions(args.engine),
    })
    start = perf_counter()
    with closing(Monitor(RocketLandingEnv(args.engine))) as env:
        model = PPO(
            "MlpPolicy", env, device="cpu", seed=args.seed_start, n_steps=1024, batch_size=64,
            gamma=0.99, tensorboard_log=str(output / "tensorboard"), verbose=1,
        )
        model.learn(total_timesteps=args.steps)
        checkpoint = output / "final.zip"
        model.save(checkpoint)
        write_json(output / "complete.json", {
            "actual_steps": model.num_timesteps, "wall_seconds": perf_counter() - start,
            "sha256": sha256(checkpoint),
        })


def episode(env, action_for, seed):
    observation, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(action_for(observation))
        steps += 1
        total += reward
        if terminated or truncated:
            return {
                "seed": seed, "return": total, "agent_steps": steps,
                "terminated": terminated, "truncated": truncated, **info,
            }


def evaluate(args):
    policies = {}
    if args.checkpoint:
        from stable_baselines3 import PPO

        model = PPO.load(args.checkpoint, device="cpu")
        policies["PPO"] = lambda obs: model.predict(obs, deterministic=True)[0]
    if not args.skip_pd:
        policies["PD"] = pd_action
    if args.ballistic:
        policies["Engine off"] = lambda obs: np.array([-1., 0., 0.], dtype=np.float32)
    if not policies:
        raise ValueError("Select at least one policy.")
    args.output.mkdir(exist_ok=False, parents=True)
    rows, report = [], []
    for condition in args.conditions:
        with closing(RocketLandingEnv(
            args.engine, condition=condition, show_window=args.show_window, movie=args.movie
        )) as env:
            for policy, action_for in policies.items():
                group = [
                    episode(env, action_for, seed)
                    for seed in range(args.seed_start, args.seed_start + args.episodes)
                ]
                impacts = [row["impact_speed"] for row in group if row["impact_speed"] is not None]
                rows.extend({"policy": policy, "condition": condition, **row} for row in group)
                report.append({
                    "policy": policy, "condition": condition, "episodes": len(group),
                    "success_rate": mean(row["success"] for row in group),
                    "mean_center_error": mean(row["center_error"] for row in group),
                    "mean_fuel_used": mean(row["fuel_used"] for row in group),
                    "mean_impact_speed": mean(impacts) if impacts else None,
                    "mean_terminal_up_dot": mean(row["up_dot"] for row in group),
                    "mean_return": mean(row["return"] for row in group),
                    "landings": sum(row["success"] for row in group),
                    "reasons": dict(Counter(row["reason"] for row in group)),
                })
                print(report[-1], flush=True)
    write_csv(args.output / "episodes.csv", rows)
    write_json(args.output / "summary.json", {
        **simulator_versions(args.engine),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "conditions": {condition: CONDITIONS[condition] for condition in args.conditions},
        "results": report,
    })


def main():
    parser = argparse.ArgumentParser(description="Run PPO or PD in the RocketRL Godot simulator.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("engine", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--seed-start", type=int, default=42 if name == "train" else 3000)
    commands.choices["train"].add_argument("--steps", type=int, required=True)
    evaluator = commands.choices["evaluate"]
    evaluator.add_argument("--checkpoint", type=Path)
    evaluator.add_argument("--episodes", type=int, default=30)
    evaluator.add_argument("--conditions", choices=tuple(CONDITIONS), nargs="+", default=tuple(CONDITIONS))
    evaluator.add_argument("--show-window", action="store_true")
    evaluator.add_argument("--skip-pd", action="store_true")
    evaluator.add_argument("--ballistic", action="store_true")
    evaluator.add_argument("--movie", type=Path, help="Record a single visible episode to a new AVI file.")
    args = parser.parse_args()
    if args.seed_start < 0 or getattr(args, "steps", 1) < 1 or getattr(args, "episodes", 1) < 1:
        parser.error("Seed must be nonnegative and budget must be positive.")
    if not args.engine.is_file() or args.output.exists():
        parser.error("Provide an engine executable and a new output directory.")
    if args.command == "evaluate" and args.movie:
        policies = int(bool(args.checkpoint)) + int(not args.skip_pd) + int(args.ballistic)
        if policies != 1 or len(args.conditions) != 1 or args.episodes != 1:
            parser.error("Movie recording requires one policy, one condition, and one episode.")
        if args.movie.suffix.lower() != ".avi" or args.movie.exists() or not args.movie.parent.is_dir():
            parser.error("Movie recording requires a new .avi file in an existing directory.")
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
