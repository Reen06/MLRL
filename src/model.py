"""
Defines the neural network architecture used for the MNIST classifier.

This file will be expanded in the next step to include a fully working model.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleClassifier(nn.Module):
    """Placeholder network architecture for the MNIST project."""

    def __init__(self) -> None:
        super().__init__()
        # The final architecture and layers will be fleshed out in the next steps.
        self.layers = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network."""
        return self.layers(inputs)
