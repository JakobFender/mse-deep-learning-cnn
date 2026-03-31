from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class DataConfig(BaseModel):
    image_size: int = 224
    num_classes: int = 10
    num_workers: int = 4
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ColorJitterConfig(BaseModel):
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    hue: float = 0


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = False
    vertical_flip: bool = False
    random_crop: bool = False
    color_jitter: bool = False
    color_jitter_config: Optional[ColorJitterConfig] = None


class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    device: str = Field(default_factory=_get_device)
    data: DataConfig
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9  # SGD only
    weight_decay: float = 0
    dropout: float = 0
    seed: int = 42
    augmentation: Optional[AugmentationConfig] = None  # No augmentation config means no augmentations
