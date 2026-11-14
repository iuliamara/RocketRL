import atexit
from contextlib import suppress
import socket
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from godot_rl.core.godot_env import GodotEnv


PROJECT = Path(__file__).resolve().parent
CONDITIONS = {
    "nominal": {}, "high": {"altitude_offset": 3}, "wide": {"offset_scale": 2.0},
    "fast": {"speed_scale": 1.7}, "heavy": {"payload": 0.3},
    "low_thrust": {"efficiency": 0.8}, "wind": {"wind": 0.65}, "low_fuel": {"fuel_fraction": 0.55},
}


class LocalGodotConnection(GodotEnv):
    """Use the upstream v0.7 transport and space negotiation with a local engine process."""

    def __init__(self, engine, show_window, movie=None):
        options = [] if show_window else ["--headless", "--disable-render-loop", "--fixed-fps", "60"]
        if movie is not None:
            options.extend(("--write-movie", str(movie.resolve()), "--fixed-fps", "60"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(15)
            self.listener = listener
            port = listener.getsockname()[1]
            self.process = subprocess.Popen([
                str(Path(engine).resolve()), "--path", str(PROJECT), *options, "--", f"--rl-port={port}",
            ])
            try:
                super().__init__(env_path="debug", port=port, convert_action_space=True)
            except BaseException:
                self.close()
                raise

    def _start_server(self):
        connection, _ = self.listener.accept()
        connection.settimeout(15)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection

    def _get_data(self):
        def receive(size):
            result = bytearray()
            while len(result) < size:
                block = self.connection.recv(size - len(result))
                if not block:
                    raise ConnectionError("Godot closed the connection.")
                result.extend(block)
            return bytes(result)

        size = int.from_bytes(receive(4), "little")
        if not 0 < size <= 65_536:
            raise ValueError(f"Invalid Godot message length: {size}")
        return receive(size).decode("utf-8")

    def close(self):
        atexit.unregister(self._close)
        if getattr(self, "connection", None) is not None:
            with suppress(OSError):
                self._send_as_json({"type": "close"})
            self.connection.close()
            self.connection = None
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=3)


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, engine, *, condition="nominal", show_window=False, movie=None):
        self.conditions = CONDITIONS[condition]
        self.render_mode = "human" if show_window or movie is not None else None
        self.bridge = LocalGodotConnection(engine, self.render_mode is not None, movie)
        self.observation_space = self.bridge.observation_space["state"]
        self.action_space = self.bridge.action_space
        self.active = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        self.bridge._send_as_json({"type": "reset", "seed": episode_seed, "conditions": self.conditions})
        result = self.bridge._get_json_dict()
        if result["type"] != "reset":
            raise RuntimeError("Godot did not acknowledge reset.")
        self.active = True
        return np.asarray(result["obs"][0]["state"], dtype=np.float32), {"seed": episode_seed}

    def step(self, action):
        if not self.active:
            raise RuntimeError("Reset the simulator before stepping.")
        observations, rewards, done, _, info = self.bridge.step(np.asarray([action], dtype=np.float32))
        truncated = bool(done[0] and info[0]["reason"] == "time_limit")
        self.active = not done[0]
        return (
            np.asarray(observations[0]["state"], dtype=np.float32), float(rewards[0]),
            bool(done[0] and not truncated), truncated, info[0],
        )

    def close(self):
        self.bridge.close()
