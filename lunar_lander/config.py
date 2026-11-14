from dataclasses import dataclass
from pathlib import Path


ENV_ID = "LunarLander-v3"
HISTORICAL_SEEDS = range(1000, 1100)
HELD_OUT_SEEDS = range(2000, 2100)
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")


@dataclass(frozen=True)
class TrainingConfig:
    reward: str = "baseline"
    seed: int = 42
    steps: int = 1_000_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    checkpoint_every: int = 102_400

    @property
    def run_id(self):
        return f"{self.reward}-s{self.seed}-n{self.steps}"
