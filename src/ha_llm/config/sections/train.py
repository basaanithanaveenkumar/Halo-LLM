from typing import Literal

from ha_llm.config.sections.common import StrictModel


class TrainConfig(StrictModel):
    steps: int | None = 300
    epochs: int | None = None
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True
    loss_type: Literal["ce", "focal", "label_smoothing"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    label_smoothing: float = 0.1
