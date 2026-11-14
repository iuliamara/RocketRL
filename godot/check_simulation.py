import argparse
from contextlib import closing
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from .environment import RocketLandingEnv
from .experiment import episode


def main():
    parser = argparse.ArgumentParser(description="Validate the RocketRL Godot and Gymnasium interface.")
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    idle = np.array([-1., 0., 0.], dtype=np.float32)
    full = np.array([1., 0., 0.], dtype=np.float32)
    with closing(RocketLandingEnv(args.engine)) as env:
        check_env(env)
        initial, _ = env.reset(seed=12)
        coast, _, _, _, coastal_info = env.step(idle)
        repeated, _ = env.reset(seed=12)
        np.testing.assert_array_equal(initial, repeated)
        thrust, _, _, _, powered_info = env.step(full)
        assert thrust[4] > coast[4], "Full thrust should increase vertical velocity relative to coasting."
        assert coastal_info["ticks"] == powered_info["ticks"] == 4
        assert powered_info["fuel_used"] > coastal_info["fuel_used"] == 0
        first = episode(env, lambda _: idle, 12)
        second = episode(env, lambda _: idle, 12)
        assert first == second
        assert first["terminated"] and not first["truncated"] and first["reason"] == "hard_contact"
        assert first["fuel_used"] == 0.0

        def hover(observation):
            mass = 1 + 0.4 * observation[12]
            vertical_speed = 8 * observation[4]
            throttle = np.clip(mass * (3.2 - 2 * vertical_speed) / 10, 0, 1)
            return np.array([2 * throttle - 1, 0., 0.], dtype=np.float32)

        cutoff = episode(env, hover, 12)
        assert cutoff["truncated"] and not cutoff["terminated"] and cutoff["reason"] == "time_limit"
        assert cutoff["ticks"] == 900 and 0.0 < cutoff["fuel_used"] < 0.4
        print(f"Gymnasium checks passed. Repeatable hard contact at tick {first['ticks']} and hover truncation at tick 900.")


if __name__ == "__main__":
    main()
