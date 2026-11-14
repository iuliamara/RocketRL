import gymnasium as gym

from .config import ENV_ID
from .rewards import LandingObjective, PotentialBasedReward, PrecisionLandingReward


def make_environment(*, render_mode=None, **options):
    """Create the evaluation environment with original rewards."""
    return gym.make(ENV_ID, render_mode=render_mode, **options)


def make_training_environment(config):
    from stable_baselines3.common.monitor import Monitor

    env = make_environment()
    if config.reward == "baseline":
        return Monitor(env)
    if config.reward == "precision":
        return Monitor(PrecisionLandingReward(env))
    if config.reward == "landing-objective":
        return Monitor(LandingObjective(env))
    if config.reward in {"potential", "potential-legacy"}:
        return Monitor(PotentialBasedReward(
            env, gamma=config.gamma, legacy_zero_on_truncation=config.reward == "potential-legacy"
        ))
    env.close()
    raise ValueError(f"Unknown reward configuration: {config.reward}")
