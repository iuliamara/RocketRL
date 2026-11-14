from collections import Counter
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass(frozen=True)
class LandingState:
    x: float
    pad_left: float
    pad_right: float
    safe_rest: bool
    both_legs: bool

    @property
    def on_pad(self):
        return self.pad_left <= self.x <= self.pad_right

    @property
    def center_error(self):
        return self.x - (self.pad_left + self.pad_right) / 2

    @property
    def pad_distance(self):
        return max(self.pad_left - self.x, self.x - self.pad_right, 0.0)

    @property
    def outcome(self):
        if not self.safe_rest:
            return "no_safe_rest"
        if not self.on_pad:
            return "off_pad"
        return "strict_landing" if self.both_legs else "bad_contact"


def landing_state(sim, terminated, truncated):
    return LandingState(
        x=float(sim.lander.position.x),
        pad_left=float(sim.helipad_x1),
        pad_right=float(sim.helipad_x2),
        safe_rest=bool(terminated and not truncated and not sim.game_over and not sim.lander.awake),
        both_legs=all(leg.ground_contact for leg in sim.legs),
    )


def end_reason(sim, observation, terminated, truncated):
    """Report both underlying termination and time limit if they coincide."""
    reasons = []
    if terminated:
        if sim.game_over:
            reasons.append("body_contact")
        if abs(observation[0]) >= 1.0:
            reasons.append("out_of_bounds")
        if not reasons:
            reasons.append("rest" if not sim.lander.awake else "other_termination")
    if truncated:
        reasons.append("time_limit")
    return "+".join(reasons)


def summarize(episodes):
    if not episodes:
        raise ValueError("Evaluation requires at least one episode.")
    returns = [row["original_return"] for row in episodes]
    outcomes = Counter(row["outcome"] for row in episodes)
    safe = [row for row in episodes if row["safe_rest"]]
    misses = [row for row in safe if row["outcome"] == "off_pad"]
    count = len(episodes)
    return {
        "episodes": count,
        "mean_original_return": mean(returns),
        "return_sd": stdev(returns) if count > 1 else None,
        "mean_steps": mean(row["steps"] for row in episodes),
        "high_scores": sum(value >= 200 for value in returns),
        "safe_rests": len(safe),
        "safe_rest_rate": len(safe) / count,
        "safe_rests_on_pad": sum(row["on_pad"] for row in safe),
        "safe_rests_both_legs": sum(row["both_legs"] for row in safe),
        "strict_landing_rate": outcomes["strict_landing"] / count,
        "outcomes": {key: outcomes[key] for key in ("strict_landing", "off_pad", "bad_contact", "no_safe_rest")},
        "termination_reasons": dict(Counter(row["end_reason"] for row in episodes)),
        "safe_rest_position_mae": mean(abs(row["center_error"]) for row in safe) if safe else None,
        "mean_engine_cost": mean(row["engine_cost"] for row in episodes),
        "off_pad": {
            "left": sum(row["center_error"] < 0 for row in misses),
            "right": sum(row["center_error"] > 0 for row in misses),
            "mean_distance": mean(row["pad_distance"] for row in misses) if misses else None,
            "max_distance": max((row["pad_distance"] for row in misses), default=None),
            "within_0.5": sum(row["pad_distance"] <= 0.5 for row in misses),
        },
    }
