"""
Evaluation utilities for the trained MNIST classifier.

This module loads the saved model weights and reports accuracy on the MNIST test
set. The dataset is downloaded automatically if it is not already present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import SimpleClassifier
from .train import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, get_dataloaders


@dataclass
class EvaluationConfig:
    """Runtime parameters for evaluation."""

    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    batch_size: int = 256  # Larger batch since we only do a forward pass.


def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Compute classification accuracy for the provided model/dataloader pair.

    Args:
        model: Network with loaded weights.
        data_loader: Loader providing MNIST test batches.
        device: Torch device used for inference (CPU by default).

    Returns:
        Accuracy as a percentage in the `[0, 100]` range.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return 100.0 * correct / max(total, 1)


def run_evaluation(config: EvaluationConfig) -> None:
    """
    Load a trained checkpoint and report test-set accuracy.

    Args:
        config: EvaluationConfig instance detailing data/checkpoint locations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {config.checkpoint_path}. "
            "Train the model first via `python -m src.train`."
        )

    # Prepare model and load weights.
    model = SimpleClassifier().to(device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # Build the dataloader (downloads test data automatically if needed).
    _, test_loader = get_dataloaders(config.data_dir, config.batch_size)

    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.2f}%")


def parse_args() -> EvaluationConfig:
    """Parse CLI arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate the MNIST classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EvaluationConfig.data_dir,
        help="Directory where MNIST is stored.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=EvaluationConfig.checkpoint_path,
        help="Path to the trained model weights file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EvaluationConfig.batch_size,
        help="Batch size used during evaluation.",
    )

    args = parser.parse_args()
    return EvaluationConfig(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    run_evaluation(parse_args())
