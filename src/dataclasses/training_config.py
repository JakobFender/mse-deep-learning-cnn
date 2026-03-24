from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class DataConfig(BaseModel):
    image_size: int = 224
    num_classes: int = 10
    num_workers: int = 4
    development_set_split: float = 0.25  # 24000 * 0.25 = 6000
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


class BaseTrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9  # SGD only
    weight_decay: float = 0
    dropout: float = 0
    seed: int = 42
    augmentation: Optional[AugmentationConfig] = None  # No augmentation config means no augmentations
