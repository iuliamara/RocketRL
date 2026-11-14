import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description="Render the retained TensorBoard episode curves.")
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    traces = defaultdict(dict)
    tags = ("rollout/ep_rew_mean", "rollout/ep_len_mean")
    for run in sorted(path for path in args.runs.iterdir() if path.is_dir()):
        for path in sorted(run.glob("events.out.tfevents.*")):
            events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
            for tag in tags:
                if tag in events.Tags()["scalars"]:
                    traces[run.name].setdefault(tag, {}).update((record.step, record.value) for record in events.Scalars(tag))
    if not traces:
        parser.error("No matching TensorBoard scalar logs.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for axis, tag, title in zip(axes, tags, ("Mean training return", "Mean episode length")):
        for name, series in traces.items():
            if tag in series:
                steps, values = zip(*sorted(series[tag].items()))
                axis.plot(steps, values, label=name.removesuffix("_1").replace("_", " "), linewidth=1.5)
        axis.set(title=title, xlabel="Training steps")
        axis.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
