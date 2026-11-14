import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROFILES = {"Baseline PPO": "#186087", "Legacy potential PPO": "#be5428"}
ACTIONS = ["idle", "left", "main", "right"]


def trajectory_metrics(frames):
    frames = frames.sort_values("step").reset_index(drop=True)
    center = (frames["pad_left"] + frames["pad_right"]) / 2
    error = frames["x"] - center
    altitude = frames["y"] - frames["pad_y"]
    contact = frames["left_contact"] | frames["right_contact"] | frames["body_contact"]
    touchdown = contact.idxmax() if contact.any() else None
    descent = frames.loc[:touchdown] if touchdown is not None else frames
    fire = frames["action"].isin([1, 3]).rolling(10, min_periods=10).sum()
    # First pre-contact ten-step window with side-engine use and decreasing outward speed.
    slowing = (frames["vx"] * error) > 0
    slowing &= frames["vx"].abs().shift(10) - frames["vx"].abs() >= 0.25
    braking = (fire >= 5) & slowing & (frames["vy"] < 0) & ~contact
    candidates = descent.index[braking.loc[descent.index]]
    correction_altitude = altitude.loc[candidates[0]] if len(candidates) else None
    low_descent = (altitude <= 6) & (frames["vy"] < 0) & ~contact
    primed = descent.index[low_descent.loc[descent.index]]
    low_descent_speed = abs(frames.loc[primed[0], "vx"]) if len(primed) else None
    result = {
        "max_abs_x_error": error.abs().max(),
        "first_contact_altitude": altitude.loc[touchdown] if touchdown is not None else None,
        "first_contact_pad_distance": max(
            frames.loc[touchdown, "pad_left"] - frames.loc[touchdown, "x"],
            frames.loc[touchdown, "x"] - frames.loc[touchdown, "pad_right"], 0,
        ) if touchdown is not None else None,
        "pre_contact_abs_vx": abs(frames.loc[touchdown - 1, "vx"]) if touchdown else None,
        "pre_contact_downward_speed": max(0, -frames.loc[touchdown - 1, "vy"]) if touchdown else None,
        "low_descent_abs_vx": low_descent_speed,
        "correction_altitude": correction_altitude,
    }
    for index, action in enumerate(ACTIONS):
        result[f"{action}_fraction"] = (frames["action"] == index).sum() / (len(frames) - 1)
    return result


def read_run(label, run):
    episodes = pd.read_csv(run / "episodes.csv").set_index("seed")
    frames = pd.read_csv(run / "trajectory.csv")
    groups = {seed: rows for seed, rows in frames.groupby("seed")}
    if set(groups) != set(episodes.index) or not episodes.index.is_unique:
        raise ValueError(f"Inconsistent episode and trajectory seeds: {run}")
    stats = pd.DataFrame.from_dict({seed: trajectory_metrics(rows) for seed, rows in groups.items()}, orient="index")
    return episodes.join(stats).assign(policy=label), groups


def sample_figure(profiles, seeds, output):
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.4 * len(seeds), 6), squeeze=False, layout="constrained")
    for column, seed in enumerate(seeds):
        for label, (episodes, groups) in profiles.items():
            rows = groups[seed]
            track = axes[0, column]
            if label == next(iter(profiles)):
                track.axvspan(rows["pad_left"].iloc[0], rows["pad_right"].iloc[0], color="#cfdfd4", alpha=0.7)
            track.plot(rows["x"], rows["y"] - rows["pad_y"], color=PROFILES[label], label=label)
            track.scatter(rows["x"].iloc[-1], (rows["y"] - rows["pad_y"]).iloc[-1], color=PROFILES[label], s=18)
            axes[1, column].plot(rows["step"] / 50, rows["vx"], color=PROFILES[label])
            style = dict(color=PROFILES[label], fontsize=8, transform=track.transAxes)
            track.text(0.02, 0.96 if label == next(iter(profiles)) else 0.87, episodes.loc[seed, "outcome"].replace("_", " "), **style)
        axes[0, column].set(title=f"Seed {seed}", xlabel="Horizontal position")
        axes[1, column].set(xlabel="Time (s)")
        axes[1, column].axhline(0, color="#8c9399", linewidth=0.75)
    axes[0, 0].set_ylabel("Height above pad (Box2D units)")
    axes[1, 0].set_ylabel("Horizontal velocity (units/s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def distribution_figure(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for axis, name, title in zip(axes, ["max_abs_x_error", "low_descent_abs_vx", "first_contact_pad_distance"],
                                 ["Peak horizontal error", "Lateral speed below 6 units", "First-contact distance to pad"]):
        for label, color in PROFILES.items():
            rows = data.loc[data["policy"] == label, name].dropna().sort_values()
            axis.step(rows, range(1, len(rows) + 1), where="post", color=color, label=label, linewidth=2)
        axis.set(xlabel=title)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Cumulative episode count")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def analyze(baseline, potential, output):
    profiles = {label: read_run(label, path) for label, path in zip(PROFILES, (baseline, potential))}
    left, right = (value[0] for value in profiles.values())
    if set(left.index) != set(right.index):
        raise ValueError("Both policies must have the same evaluation seeds.")
    left_groups, right_groups = (value[1] for value in profiles.values())
    for seed in left.index:
        columns = ["x", "y", "vx", "vy", "angle", "pad_left", "pad_right", "pad_y"]
        before = left_groups[seed].iloc[0]
        after = right_groups[seed].iloc[0]
        if before["step"] != 0 or after["step"] != 0 or not before[columns].equals(after[columns]):
            raise ValueError(f"Initial states differ at seed {seed}.")
    data = pd.concat([left, right]).rename_axis("seed")
    output.mkdir(exist_ok=False, parents=True)
    data.to_csv(output / "episode_diagnostics.csv")
    for name, groups in (("overall", ["policy"]), ("by_outcome", ["policy", "outcome"])):
        metrics = data.columns[data.columns.get_loc("max_abs_x_error"):].drop("policy")
        grouped = data.groupby(groups, sort=False)
        table = grouped[list(metrics)].agg(["count", "mean", "median"])
        table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
        table.insert(0, "episodes", grouped.size())
        table.to_csv(output / f"{name}.csv", float_format="%.5f")
    pd.crosstab(left["outcome"], right["outcome"], rownames=["baseline"], colnames=["potential"]).to_csv(output / "paired_outcomes.csv")
    data.groupby(["policy", "end_reason"]).size().to_csv(output / "termination_reasons.csv", header=["episodes"])
    seeds = [left.index.min()]
    for baseline_result, potential_result in (("strict_landing", "off_pad"), ("strict_landing", "strict_landing"), ("off_pad", "strict_landing")):
        matching = sorted(set(left.index[left["outcome"] == baseline_result]) & set(right.index[right["outcome"] == potential_result]))
        seeds.extend(seed for seed in matching[:1] if seed not in seeds)
    sample_figure(profiles, seeds, output / "paired_trajectories.png")
    distribution_figure(data, output / "trajectory_distributions.png")


def main():
    parser = argparse.ArgumentParser(description="Compare two matched LunarLander telemetry runs.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("potential", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.baseline, args.potential, args.output)


if __name__ == "__main__":
    main()
