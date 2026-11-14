import argparse
import subprocess
from contextlib import closing
from pathlib import Path

from .environment import make_environment
from .rollout import run_episode


def record(checkpoint, output, ffmpeg, seeds):
    from PIL import Image
    from stable_baselines3 import PPO

    output.mkdir(parents=True, exist_ok=False)
    model = PPO.load(checkpoint, device="cpu")
    with closing(make_environment(render_mode="rgb_array")) as env:
        for seed in seeds:
            screenshot = output / f"seed-{seed}.png"
            clip = output / f"seed-{seed}.mp4"
            with subprocess.Popen([
                str(ffmpeg), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", "600x400", "-framerate", "50", "-i", "-",
                "-vf", "tpad=stop_mode=clone:stop_duration=1", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip),
            ], stdin=subprocess.PIPE) as encoder:
                def write_frame(environment):
                    frame = environment.render()
                    encoder.stdin.write(frame.tobytes())
                result, _ = run_episode(env, model, seed, on_frame=write_frame)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f"Encoding failed for seed {seed}")
            Image.fromarray(env.render()).save(screenshot)
            print(f"{clip} | {result['outcome']} | {result['steps']} steps")


def main():
    parser = argparse.ArgumentParser(description="Render deterministic LunarLander runs as H.264 clips.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique and nonnegative.")
    record(args.checkpoint, args.output, args.ffmpeg, args.seeds)


if __name__ == "__main__":
    main()
