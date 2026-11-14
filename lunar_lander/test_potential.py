if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


import unittest
from types import SimpleNamespace

import gymnasium as gym

from .metrics import end_reason, landing_state, summarize
from .rewards import LandingObjective, PotentialBasedReward, PrecisionLandingReward
from .study import stats, summarize_study


class ControlledLandings(gym.Env):
    def __init__(self, *, terminated=False, truncated=False, x=10, awake=False, crashed=False, legs=(True, True)):
        self.terminated = terminated
        self.truncated = truncated
        self.helipad_x1, self.helipad_x2 = 8, 12
        self.lander = SimpleNamespace(position=SimpleNamespace(x=x), awake=awake)
        self.game_over = crashed
        self.legs = [SimpleNamespace(ground_contact=contact) for contact in legs]

    def reset(self, **kwargs):
        return [0.5, 0.0, -0.2], {"reset": True}

    def step(self, action):
        return [0.2, 0.0, 0.4, -0.2], 5.0, self.terminated, self.truncated, {"step": True}


class RewardTests(unittest.TestCase):
    def test_shaping_preserves_cutoff_potential_and_flags(self):
        for terminated, truncated, legacy, expected in (
            (False, False, False, 9.0),
            (False, True, False, 9.0),
            (False, True, True, 17.0),
            (True, False, False, 17.0),
            (True, True, False, 17.0),
        ):
            with self.subTest(terminated=terminated, truncated=truncated, legacy=legacy):
                env = PotentialBasedReward(ControlledLandings(
                    terminated=terminated, truncated=truncated
                ), gamma=1.0, legacy_zero_on_truncation=legacy)
                self.assertEqual(env.reset()[1], {"reset": True})
                self.assertEqual(env.step(0), ([0.2, 0.0, 0.4, -0.2], expected, terminated, truncated, {"step": True}))

    def test_discounted_shaping_telescopes_and_reset_reinitializes(self):
        base = ControlledLandings()
        env = PotentialBasedReward(base, gamma=0.9)
        env.reset()
        first = env.step(0)[1] - 5.0
        self.assertAlmostEqual(first, 0.9 * -8 - (-12))
        base.terminated = True
        second = env.step(0)[1] - 5.0
        self.assertAlmostEqual(first + 0.9 * second, 12.0)
        env.reset()
        self.assertAlmostEqual(env.step(0)[1], 17.0)

    def test_precision_uses_safe_rest_and_inclusive_pad_bounds(self):
        cases = [
            ({"terminated": True, "x": 8, "legs": (True, False)}, 55),
            ({"terminated": True, "x": 12}, 55),
            ({"terminated": True, "x": 12.1}, -45),
            ({"terminated": True, "crashed": True}, 5),
            ({"terminated": True, "awake": True}, 5),
            ({"terminated": True, "truncated": True}, 5),
            ({"truncated": True}, 5),
        ]
        for flags, reward in cases:
            with self.subTest(flags=flags):
                self.assertEqual(PrecisionLandingReward(ControlledLandings(**flags)).step(0)[1], reward)

    def test_explicit_objective_charges_drift_engine_use_and_strict_outcome(self):
        for flags, action, expected in (
            ({}, 2, 4.65),
            ({"terminated": True}, 1, 54.92),
            ({"terminated": True, "legs": (True, False)}, 0, -45.05),
            ({"truncated": True}, 0, -45.05),
        ):
            with self.subTest(flags=flags, action=action):
                env = LandingObjective(ControlledLandings(**flags))
                self.assertAlmostEqual(env.step(action)[1], expected)


class MetricsTests(unittest.TestCase):
    def test_landing_outcomes(self):
        cases = [
            ({"x": 8}, "strict_landing", 0.0),
            ({"x": 7.75, "legs": (True, False)}, "off_pad", 0.25),
            ({"x": 12.5}, "off_pad", 0.5),
            ({"legs": (True, False)}, "bad_contact", 0.0),
            ({"awake": True}, "no_safe_rest", 0.0),
            ({"crashed": True}, "no_safe_rest", 0.0),
        ]
        for flags, outcome, distance in cases:
            with self.subTest(flags=flags):
                state = landing_state(ControlledLandings(**flags), True, False)
                self.assertEqual((state.outcome, state.pad_distance), (outcome, distance))
        self.assertFalse(landing_state(ControlledLandings(), True, True).safe_rest)

    def test_termination_diagnostics(self):
        env = ControlledLandings(crashed=True)
        self.assertEqual(end_reason(env, [1.0], True, True), "body_contact+out_of_bounds+time_limit")
        self.assertEqual(end_reason(env, [0.0], False, True), "time_limit")
        self.assertEqual(end_reason(ControlledLandings(), [0.0], True, False), "rest")

    def test_summary_denominators_and_miss_counts(self):
        rows = []
        for x, legs, reward in ((10, True, 200), (8, False, 210), (12.5, True, 180), (7, True, 190)):
            state = landing_state(ControlledLandings(x=x, legs=(True, legs)), True, False)
            rows.append({
                "steps": 12, "original_return": reward, "outcome": state.outcome,
                "safe_rest": state.safe_rest, "on_pad": state.on_pad, "both_legs": state.both_legs,
                "center_error": state.center_error, "pad_distance": state.pad_distance,
                "end_reason": "rest", "engine_cost": 1.5,
            })
        summary = summarize(rows)
        self.assertEqual(summary["high_scores"], 2)
        self.assertEqual(summary["mean_original_return"], 195)
        self.assertEqual(summary["strict_landing_rate"], 0.25)
        self.assertEqual(summary["safe_rests_on_pad"], 2)
        self.assertEqual(summary["off_pad"], {
            "left": 1, "right": 1, "mean_distance": 0.75, "max_distance": 1, "within_0.5": 1,
        })
        self.assertIsNone(summarize(rows[:1])["return_sd"])


class StudyTests(unittest.TestCase):
    def test_paired_differences_use_matching_training_seeds(self):
        def row(recipe, seed, value):
            return {
                "recipe": recipe, "training_seed": seed, "split": "historical",
                **{key: value for key in (
                    "mean_original_return", "strict_landing_rate", "safe_rest_rate",
                    "safe_rest_position_mae", "mean_steps", "mean_engine_cost",
                )},
            }
        _, contrasts = summarize_study(
            [row("baseline", 42, 1), row("baseline", 43, 3), row("potential", 43, 5)],
            ["baseline", "potential"], ["historical"],
        )
        self.assertEqual(contrasts[0]["training_seeds"], 1)
        self.assertEqual(contrasts[0]["mean"], 2)
        self.assertIsNone(contrasts[0]["ci95_low"])
        self.assertEqual(stats([0.5, 0.5, 0.5])["ci95_high"], 0.5)


if __name__ == "__main__":
    unittest.main()
