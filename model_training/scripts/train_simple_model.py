"""
Train a tiny bag-of-words classifier with vanilla PyTorch.

The script expects `model_training/scripts/prepare_simple_dataset.py` to have
produced `train.jsonl`, `val.jsonl`, and `simple_vocab.json` inside
`model_training/data/`.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass
class Record:
    text: str
    label: str


def load_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(Record(text=payload["text"], label=payload["label"]))
    return records


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_label_mapping(records: Iterable[Record]) -> Dict[str, int]:
    labels = sorted({record.label for record in records})
    return {label: idx for idx, label in enumerate(labels)}


def vectorize(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for token in TOKEN_PATTERN.findall(text.lower()):
        if token in vocab:
            vec[vocab[token]] += 1.0
    return vec


class OutcomeDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, records: List[Record], vocab: Dict[str, int], label_map: Dict[str, int]) -> None:
        self.records = records
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        features = vectorize(record.text, self.vocab)
        label_idx = self.label_map[record.label]
        return features, label_idx


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def build_model(
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    hidden_size = max(8, hidden_size)
    hidden_layers = max(1, hidden_layers)
    dropout = max(0.0, min(dropout, 0.9))

    for idx in range(hidden_layers):
        in_features = input_dim if idx == 0 else hidden_size
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_size, output_dim))
    return nn.Sequential(*layers)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    best_acc = 0.0
    best_state = model.state_dict()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | train_acc={train_acc:.2%} | val_acc={val_acc:.2%}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple PyTorch classifier over outcome texts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("model_training") / "data",
        help="Location of train.jsonl/val.jsonl and simple_vocab.json (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_training") / "models" / "simple_model.pt",
        help="File to save the best model weights (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Width of each hidden layer (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability applied after each hidden layer (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the Adam optimizer (default: %(default)s)",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    vocab_path = args.data_dir / "simple_vocab.json"

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    vocab = load_vocab(vocab_path)
    label_map = build_label_mapping(train_records + val_records)

    train_dataset = OutcomeDataset(train_records, vocab, label_map)
    val_dataset = OutcomeDataset(val_records, vocab, label_map)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        input_dim=len(vocab),
        output_dim=len(label_map),
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, history = train(model, train_loader, val_loader, args.epochs, optimizer, device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "label_map": label_map,
            "history": history,
            "config": {
                "hidden_size": args.hidden_size,
                "hidden_layers": args.hidden_layers,
                "dropout": args.dropout,
                "input_dim": len(vocab),
                "output_dim": len(label_map),
            },
        },
        args.checkpoint,
    )

    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved best model to {args.checkpoint}")


if __name__ == "__main__":
    main()

