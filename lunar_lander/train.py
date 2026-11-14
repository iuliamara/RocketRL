if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


import argparse
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from .config import ARTIFACT_DIR, ENV_ID, TrainingConfig
from .environment import make_training_environment
from .io import lunar_versions, sha256, write_json


REWARDS = ("baseline", "precision", "potential", "potential-legacy", "landing-objective")


def train(config, output, *, resume=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    if config.reward not in REWARDS:
        raise ValueError(f"Unknown reward: {config.reward}")
    if resume and config.reward != "baseline":
        raise ValueError("Checkpoint continuation is supported for the baseline recipe.")
    source = {"path": str(resume), "sha256": sha256(resume)} if resume else None
    with closing(make_training_environment(config)) as env:
        log_dir = str(output / "tensorboard")
        if resume:
            model = PPO.load(resume, env=env, device="cpu", tensorboard_log=log_dir)
            for key in ("seed", "gamma", "learning_rate", "n_steps", "batch_size"):
                if getattr(model, key) != getattr(config, key):
                    raise ValueError(f"Resume checkpoint {key} differs from the requested configuration.")
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=config.learning_rate, n_steps=config.n_steps,
                batch_size=config.batch_size, gamma=config.gamma, seed=config.seed,
                device="cpu", tensorboard_log=log_dir, verbose=1,
            )
        output.mkdir(parents=True, exist_ok=False)
        starting_steps = model.num_timesteps
        write_json(output / "config.json", {
            **asdict(config), "run_id": output.name, "environment": ENV_ID,
            "resume": source, "starting_steps": starting_steps, "versions": lunar_versions(),
        })
        callback = CheckpointCallback(
            save_freq=config.checkpoint_every, save_path=output / "checkpoints", name_prefix="ppo"
        ) if config.checkpoint_every else None
        model.learn(
            total_timesteps=config.steps, reset_num_timesteps=resume is None,
            tb_log_name=config.reward, callback=callback,
        )
        checkpoint = output / "final.zip"
        model.save(checkpoint)
        write_json(output / "complete.json", {
            "checkpoint": checkpoint.name,
            "sha256": sha256(checkpoint),
            "starting_steps": starting_steps,
            "actual_steps": model.num_timesteps,
        })
    return checkpoint


def main(argv=None, *, default_reward="baseline", default_steps=300_000, default_resume=None):
    parser = argparse.ArgumentParser(description="Train PPO and save a new experiment directory.")
    parser.add_argument("--reward", choices=REWARDS, default=default_reward)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=102_400)
    parser.add_argument("--resume", type=Path, default=default_resume)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seed < 0 or args.checkpoint_every < 0:
        parser.error("Steps must be positive. Seed and checkpoint interval must be nonnegative.")
    config = TrainingConfig(
        reward=args.reward, steps=args.steps, seed=args.seed, checkpoint_every=args.checkpoint_every
    )
    run_id = config.run_id
    if args.resume:
        run_id += f"-from-{args.resume.stem}"
    output = args.output or ARTIFACT_DIR / "train" / run_id
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --output.")
    print(train(config, output, resume=args.resume))


if __name__ == "__main__":
    main()
