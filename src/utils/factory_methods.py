from typing import Optional, Callable

from torch.nn import Module
from torch.optim import Optimizer, Adam, SGD, RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms.v2 import Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, \
    ColorJitter
from torchvision.transforms.v2 import Transform

from src.dataclasses.training_config import TrainingConfig
from src.model.cnn import CNN
from src.utils.constants import TRAIN_SET_FOLDER, VALIDATE_SET_FOLDER


def get_loaders(
        config: TrainingConfig,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Provides data loaders for training and validation datasets.

    This function creates PyTorch DataLoader objects for both training and validation
    datasets using the specified configurations and optional transformations.

    Args:
        config (TrainingConfig): Configuration object specifying training parameters,
            including batch size and data loading specifics.
        train_transform (Optional[Callable]): Transformation function to be applied
            to the training dataset. Defaults to None.
        val_transform (Optional[Callable]): Transformation function to be applied
            to the validation dataset. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the DataLoader for the
        training dataset as the first element and the DataLoader for the validation
        dataset as the second element.
    """

    train_ds = ImageFolder(root=TRAIN_SET_FOLDER, transform=train_transform)
    val_ds = ImageFolder(root=VALIDATE_SET_FOLDER, transform=val_transform)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    return train_dl, val_dl


def get_transforms(config: TrainingConfig) -> tuple[Compose, Compose]:
    """
    Generates and returns image transformation pipelines for training and validation datasets.

    The method creates a list of transformations to preprocess input data for a deep learning
    model. Initial transformations, such as tensor conversion and resizing, are applied
    to both training and validation data. Additional augmentations like random flips,
    crops, and color jittering are applied to the training data only, depending on the
    configuration options passed in the `config` parameter.

    Args:
        config (TrainingConfig): Configuration object containing augmentation settings
            and normalization parameters for both training and validation datasets.

    Returns:
        tuple[Compose, Compose]: A tuple containing two `Compose` objects, one for training
        transformations and one for validation transformations.
    """

    val_transforms: list[Transform] = [
        ToTensor(),
        Resize((224, 224)),  # Theoretically not necessary if we assume images to already be 224x224
    ]

    train_transforms: list[Transform] = val_transforms.copy()

    if config.augmentation:
        if config.augmentation.horizontal_flip:
            train_transforms.append(RandomHorizontalFlip())

        if config.augmentation.vertical_flip:
            train_transforms.append(RandomVerticalFlip())

        if config.augmentation.random_crop:
            train_transforms.append(RandomCrop(224))

        if config.augmentation.color_jitter:
            train_transforms.append(ColorJitter(
                brightness=config.augmentation.color_jitter_config.brightness,
                contrast=config.augmentation.color_jitter_config.contrast,
                saturation=config.augmentation.color_jitter_config.saturation,
                hue=config.augmentation.color_jitter_config.hue,
            ))

    val_transforms.append(Normalize(mean=config.data.mean, std=config.data.std))
    train_transforms.append(Normalize(mean=config.data.mean, std=config.data.std))

    return Compose(train_transforms), Compose(val_transforms)


def get_optimizer(config: TrainingConfig, model: Module) -> Optimizer:
    """
    Selects and returns the optimizer as specified by the given training configuration. This function
    supports multiple optimizer types such as Adam, SGD, and RMSprop and configures them with parameters
    provided in the configuration and model. If an unsupported optimizer type is specified, it raises
    a ValueError.

    Args:
        config (TrainingConfig): The training configuration object specifying optimizer settings, such
            as the optimizer type, learning rate, momentum, and weight decay.
        model (Module): The model whose parameters will be optimized, typically an instance of a neural
            network module.

    Returns:
        Optimizer: The selected optimizer configured with the parameters from the given configuration
            and model.

    Raises:
        ValueError: If an unsupported optimizer type is specified in the configuration.
    """
    if config.optimizer == "adam":
        return Adam(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "rmsprop":
        return RMSprop(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_model(config: TrainingConfig) -> CNN:
    """
    Instantiates a CNN from a TrainingConfig.

    Args:
        config: Training configuration containing model and data parameters.

    Returns:
        CNN model moved to the device specified in config.
    """
    model = CNN(
        num_classes=config.data.num_classes,
        channels=config.model.channels,
        fc_hidden_size=config.model.fc_hidden_size,
        kernel_size=config.model.kernel_size,
        pool_size=config.model.pool_size,
        dropout_p=config.dropout,
        input_size=config.data.image_size,
    )
    return model.to(config.device)
