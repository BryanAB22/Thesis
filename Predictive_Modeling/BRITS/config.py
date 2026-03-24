from dataclasses import dataclass, field
from typing import List


@dataclass
class BritsConfig:
    dataset_dir: str

    feat_cols: List[str] = field(default_factory=lambda: [
        "HR", "O2Sat", "Temp", "SBP", "DBP", "Resp","MAP"
    ])

    out_dir: str = "artifacts/brits"

    seed: int = 123
    T: int = 48
    keep_last: bool = True

    train_frac: float = 0.80
    normalize: bool = True

    hidden_size: int = 64
    consistency_weight: float = 0.1

    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5

    holdout_frac: float = 0.10
    num_workers: int = 0

    save_every: int = 1