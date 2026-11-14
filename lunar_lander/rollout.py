from collections import Counter
from contextlib import closing
from itertools import chain

from .environment import make_environment
from .metrics import end_reason, landing_state


def _frame(sim, step, action, reward):
    body = sim.lander
    return {
        "step": step,
        "action": action,
        "reward": reward,
        "x": float(body.position.x),
        "y": float(body.position.y),
        "vx": float(body.linearVelocity.x),
        "vy": float(body.linearVelocity.y),
        "angle": float(body.angle),
        "angular_velocity": float(body.angularVelocity),
        "left_contact": bool(sim.legs[0].ground_contact),
        "right_contact": bool(sim.legs[1].ground_contact),
        "body_contact": bool(sim.game_over),
        "pad_left": float(sim.helipad_x1),
        "pad_right": float(sim.helipad_x2),
        "pad_y": float(sim.helipad_y),
    }


def run_episode(env, model, seed, *, telemetry=False, on_frame=None):
    obs, _ = env.reset(seed=seed)
    sim = env.unwrapped
    frames = [_frame(sim, 0, None, None)] if telemetry else []
    if on_frame:
        on_frame(env)
    actions = Counter()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        prediction, _ = model.predict(obs, deterministic=True)
        action = int(prediction)
        obs, reward, terminated, truncated, _ = env.step(action)
        actions[action] += 1
        steps += 1
        total_reward += float(reward)
        if telemetry:
            frames.append(_frame(sim, steps, action, reward))
        if on_frame:
            on_frame(env)

    state = landing_state(sim, terminated, truncated)
    episode = {
        "seed": seed,
        "steps": steps,
        "original_return": total_reward,
        "outcome": state.outcome,
        "end_reason": end_reason(sim, obs, terminated, truncated),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "safe_rest": state.safe_rest,
        "on_pad": state.on_pad,
        "both_legs": state.both_legs,
        "final_x": state.x,
        "center_error": state.center_error,
        "pad_distance": state.pad_distance,
        "idle_steps": actions[0],
        "left_engine_steps": actions[1],
        "main_engine_steps": actions[2],
        "right_engine_steps": actions[3],
        "engine_cost": 0.3 * actions[2] + 0.03 * (actions[1] + actions[3]),
    }
    return episode, [{"seed": seed, **frame} for frame in frames]


def evaluate_checkpoint(checkpoint, seeds, *, telemetry=False, render_mode=None):
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint, device="cpu")
    with closing(make_environment(render_mode=render_mode)) as env:
        results = [run_episode(env, model, seed, telemetry=telemetry) for seed in seeds]
    episodes = [episode for episode, _ in results]
    frames = list(chain.from_iterable(frames for _, frames in results))
    return model.seed, episodes, frames
