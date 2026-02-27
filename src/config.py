"""Configuration and hyperparameters for training."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Training configuration with reproducibility seeds."""

    # Data
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model
    num_classes: int = 10
    dropout: float = 0.3

    # Training
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    seed: int = 42

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
