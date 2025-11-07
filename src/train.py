"""
Training entry point for the MNIST classifier.

Running this module will automatically download MNIST (to the `data/` folder),
build the model, and train it for a few quick epochs on CPU. A checkpoint with
the learned weights is written to `data/mnist_cnn.pt` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import SimpleClassifier

# Constants that keep file and folder names in one place.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHECKPOINT = DEFAULT_DATA_DIR / "mnist_cnn.pt"


@dataclass
class TrainingConfig:
    """Container for the most important training hyperparameters."""

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    seed: int = 42


def get_dataloaders(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch dataloaders for MNIST training and testing data.

    Args:
        data_dir: Directory where MNIST will be downloaded/cached.
        batch_size: Number of samples per batch.

    Returns:
        A tuple of `(train_loader, test_loader)`.
    """
    # Standard normalization constants for MNIST.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,  # <-- Automatically fetches the dataset when missing.
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,  # Safe to call; it is a no-op if files already exist.
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Compute accuracy for the provided dataloader.

    Args:
        model: The network to evaluate.
        data_loader: Loader that yields `(input, target)` batches.
        device: Torch device (`cpu` by default).

    Returns:
        Accuracy as a floating-point percentage (0-100).
    """
    model.eval()
    total = 0
    correct = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return 100.0 * correct / max(total, 1)


def train(config: TrainingConfig) -> None:
    """
    Execute the training loop using the provided configuration.

    Args:
        config: TrainingConfig instance with hyperparameters and paths.
    """
    # Ensure deterministic-ish behaviour for reproducibility.
    torch.manual_seed(config.seed)

    # Stick to CPU by default but allow CUDA if the user has it available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataloaders (MNIST is downloaded automatically here).
    train_loader, test_loader = get_dataloaders(config.data_dir, config.batch_size)

    # Initialize the network and optimizer.
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == targets).sum().item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * running_correct / total_samples
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"- loss: {epoch_loss:.4f} "
            f"- train_acc: {epoch_acc:.2f}% "
            f"- test_acc: {test_acc:.2f}%"
        )

    # Persist the trained weights so the evaluation script can reload them.
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
        },
        config.checkpoint_path,
    )
    print(f"Saved checkpoint to {config.checkpoint_path.resolve()}")


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a TrainingConfig instance."""
    parser = argparse.ArgumentParser(description="Train the MNIST classifier.")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=TrainingConfig.data_dir,
        help="Directory where MNIST will be stored.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=TrainingConfig.checkpoint_path,
        help="Where to save the trained model weights.",
    )
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)

    args = parser.parse_args()
    return TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    train(parse_args())
