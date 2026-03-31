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

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        # Convolutional block 1
        # Input:  (batch, 3, 128, 128)  — 3 RGB channels
        # Output: (batch, 32, 64, 64)   — after conv + 2x2 max-pool
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolutional block 2
        # Input:  (batch, 32, 64, 64)
        # Output: (batch, 64, 32, 32)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64→32
        )

        # Convolutional block 3
        # Input:  (batch, 64, 32, 32)
        # Output: (batch, 128, 16, 16)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32→16
        )

        # Fully connected classifier
        # Flattened feature vector: 128 channels × 16 × 16 = 32768
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 128, 16, 16) → (batch, 32768)
            nn.Linear(128 * 16 * 16, 512),  # Fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Think I should leave this out for now and use it later in the regularization?
            nn.Linear(512, num_classes),  # Output one logit per class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: run input through conv blocks then classifier."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x  # Raw logits — CrossEntropyLoss handles softmax internally
