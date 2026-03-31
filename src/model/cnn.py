import torch
from torch import nn


class CNN(nn.Module):
    """
    A 3-layer Convolutional Neural Network.

    Architecture:
        Conv Layer 1 → BatchNorm → ReLU → MaxPool
        Conv Layer 2 → BatchNorm → ReLU → MaxPool (Not sure if we decided on doing this or not?)
        Conv Layer 3 → BatchNorm → ReLU → MaxPool
        Flatten → Fully Connected → Dropout → Output (10 classes)
    """

    def __init__(
            self,
            num_classes: int = 10,
            channels: tuple[int, int, int] = (32, 64, 128),
            fc_hidden_size: int = 512,
            dropout_p: float = 0.5,
            kernel_size: int = 3,
            pool_size: int = 2,
            input_size: int = 128,
    ):
        super(CNN, self).__init__()

        c1, c2, c3 = channels
        padding = kernel_size // 2  # Preserve spatial dims before pooling

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

        pooled_size = input_size // (pool_size ** 3)
        flat_size = c3 * pooled_size * pooled_size

        classifier_layers: list = [
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden_size),
            nn.ReLU(inplace=True),
        ]

        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(fc_hidden_size, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: run input through conv blocks then classifier."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x  # Raw logits — CrossEntropyLoss handles softmax internally
