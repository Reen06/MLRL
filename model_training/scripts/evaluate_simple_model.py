"""
Evaluate the simple PyTorch classifier on the validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from train_simple_model import (
    OutcomeDataset,
    Record,
    accuracy,
    build_model,
    load_jsonl,
    load_vocab,
)


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the simple outcome classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("model_training") / "data",
        help="Directory containing val.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_training") / "models" / "simple_model.pt",
        help="Checkpoint path saved by train_simple_model.py (default: %(default)s)",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    vocab: Dict[str, int] = checkpoint["vocab"]
    label_map: Dict[str, int] = checkpoint["label_map"]

    val_records: List[Record] = load_jsonl(args.data_dir / "val.jsonl")
    missing = {record.label for record in val_records if record.label not in label_map}
    if missing:
        raise ValueError(
            f"Validation labels {sorted(missing)} were not seen during training. "
            "Regenerate the dataset or retrain with updated config."
        )

    dataset = OutcomeDataset(val_records, vocab, label_map)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = checkpoint.get("config") or {}
    model = build_model(
        input_dim=len(vocab),
        output_dim=len(label_map),
        hidden_size=config.get("hidden_size", 256),
        hidden_layers=config.get("hidden_layers", 2),
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)

    val_accuracy = accuracy(model, loader, device)
    result = {"validation_accuracy": val_accuracy, "num_examples": len(dataset)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

