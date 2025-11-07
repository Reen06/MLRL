"""
Neural network architecture used for the MNIST classifier.

The model intentionally stays small so it trains quickly on CPU while still
showing the typical structure of a convolutional neural network (CNN).
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleClassifier(nn.Module):
    """
    Tiny CNN that maps 28x28 grayscale digits to 10 output logits.

    The architecture uses two convolutional layers to capture spatial patterns,
    followed by a lightweight fully connected head. Dropout is included to
    mitigate overfitting without making training unstable.
    """

    def __init__(self) -> None:
        super().__init__()

        # Convolutional "feature extractor" portion of the network.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        # Fully connected classification head. The flattened feature map coming
        # out of the convolutional stack has shape (batch, 64 * 12 * 12).
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 12 * 12, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the network.

        Args:
            inputs: A batch of images shaped `(batch_size, 1, 28, 28)`.

        Returns:
            Raw, unnormalized class logits of shape `(batch_size, 10)`.
        """
        features = self.feature_extractor(inputs)
        logits = self.classifier(features)
        return logits
