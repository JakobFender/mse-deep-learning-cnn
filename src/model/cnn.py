import torch
from torch import nn


class CNN(nn.Module):
    """
    A configurable Convolutional Neural Network.

    Architecture:
        [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d] * N
        -> Flatten -> Linear -> ReLU -> (Dropout) -> Linear

    Args:
        num_classes: Number of output classes.
        channels: Output channels per conv block. Its length defines
                         the number of blocks (e.g. (32, 64, 128) -> 3 blocks).
        fc_hidden_size: Number of units in the FC hidden layer.
        dropout_p: Dropout probability (0.0 disables dropout).
        kernel_size: Convolutional kernel size (applied to all blocks).
        pool_size: Max-pool kernel size and stride (applied to all blocks).
        input_size: Spatial size of the input (assumed square, e.g. 128).
    """

    def __init__(
            self,
            num_classes: int = 10,
            channels: tuple[int, ...] = (32, 64, 128),
            fc_hidden_size: int = 512,
            dropout_p: float = 0.5,
            kernel_size: int = 3,
            pool_size: int = 2,
            input_size: int = 128,
    ):
        super(CNN, self).__init__()

        assert len(channels) >= 1, "Must have at least one conv block."
        assert input_size // (pool_size ** len(channels)) >= 1, (
            f"Too many pooling layers: spatial size collapses to 0 with "
            f"input_size={input_size}, pool_size={pool_size}, "
            f"num_blocks={len(channels)}."
        )

        padding = kernel_size // 2  # "same" spatial dims before pooling

        in_ch = 3
        blocks = []
        for out_ch in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            ))
            in_ch = out_ch

        self.conv_blocks = nn.ModuleList(blocks)

        pooled_size = input_size // (pool_size ** len(channels))
        flat_size = channels[-1] * pooled_size * pooled_size

        if flat_size >= 100_000:
            print(f"WARNING: Final flat layer is {flat_size}. This results in a very large parameter count.")

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
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the model, expected to have dimensions suitable
                for the first convolutional block.

        Returns:
            torch.Tensor: Output tensor containing raw logits representing class scores.
        """
        for block in self.conv_blocks:
            x = block(x)
        x = self.classifier(x)
        return x
