"""
Generate a very small text classification dataset from the configured outcomes.

Each training example corresponds to one of the curated examples stored in
`config/outcomes/*.json`. The script tokenizes text with a simple regex
split, builds a case-insensitive vocabulary, and writes train/validation
splits to JSONL files that the simple PyTorch trainer can consume.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
DEFAULT_SEED = 2025
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_MIN_VAL_PER_LABEL = 1
DEFAULT_AUGMENT_FACTOR = 0


@dataclass
class Example:
    text: str
    label: str

    def to_dict(self) -> Dict[str, str]:
        return {"text": self.text, "label": self.label}


@dataclass
class OutcomeData:
    outcome_id: str
    label: str
    synonyms: List[str]
    spell_variants: List[str]
    teach_hint: Optional[str]
    examples: List[Example]


def read_outcomes(config_dir: Path) -> List[OutcomeData]:
    outcomes: List[OutcomeData] = []
    outcomes_dir = config_dir / "outcomes"
    if not outcomes_dir.exists():
        raise FileNotFoundError(f"Outcomes directory not found: {outcomes_dir}")

    for file_path in sorted(outcomes_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as handle:
            outcome = json.load(handle)

        outcome_id = outcome.get("id")
        if not outcome_id:
            continue

        base_examples: List[Example] = []
        for item in outcome.get("examples", []):
            text = (item or {}).get("text")
            if text:
                base_examples.append(Example(text=text.strip(), label=outcome_id))

        outcomes.append(
            OutcomeData(
                outcome_id=outcome_id,
                label=outcome.get("label", outcome_id),
                synonyms=[s.strip().lower() for s in outcome.get("synonyms", []) if s.strip()],
                spell_variants=[s.strip().lower() for s in outcome.get("spell_variants", []) if s.strip()],
                teach_hint=outcome.get("teach_hint"),
                examples=base_examples,
            )
        )

    if not outcomes:
        raise ValueError("No outcomes discovered in config/outcomes.")
    return outcomes


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(examples: Iterable[Example], min_count: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for example in examples:
        for token in set(tokenize(example.text)):
            counts[token] = counts.get(token, 0) + 1
    vocab = {token: idx for idx, (token, count) in enumerate(sorted(counts.items())) if count >= min_count}
    return vocab


def stratified_split(
    examples: List[Example],
    train_fraction: float,
    seed: int,
    min_val_per_label: int,
) -> tuple[List[Example], List[Example]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Example]] = defaultdict(list)
    for example in examples:
        buckets[example.label].append(example)

    train: List[Example] = []
    val: List[Example] = []

    for label, bucket in buckets.items():
        rng.shuffle(bucket)
        size = len(bucket)
        if size == 1:
            train.extend(bucket)
            continue
        desired_val = int(round(size * (1 - train_fraction)))
        desired_val = max(min_val_per_label, desired_val)
        desired_val = min(desired_val, size - 1)
        val.extend(bucket[:desired_val])
        train.extend(bucket[desired_val:])

    if not val and train:
        val.append(train.pop())

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, rows: Iterable[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def write_vocab(path: Path, vocab: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, indent=2, ensure_ascii=False)


def _deterministic_seed(outcome_id: str, seed: int) -> int:
    digest = md5(outcome_id.encode("utf-8")).hexdigest()
    base = int(digest[:8], 16)
    return (base + seed) % (2**32 - 1)


def generate_augmented_examples(
    outcome: OutcomeData,
    augment_factor: int,
    seed: int,
) -> List[Example]:
    if augment_factor <= 0:
        return []

    rng = random.Random(_deterministic_seed(outcome.outcome_id, seed))
    templates = []

    if outcome.synonyms:
        templates.append(lambda syn: f"This request is about {syn}.")
        templates.append(lambda syn: f"Can you help with {syn}?")
    if outcome.spell_variants:
        templates.append(lambda variant: f"I think it's spelled '{variant}', please assist.")
    if outcome.teach_hint:
        templates.append(lambda _: f"{outcome.teach_hint}")
    templates.append(lambda _: f"Handle the {outcome.label.lower()} workflow.")
    if outcome.examples:
        exemplar_texts = [ex.text for ex in outcome.examples]
    else:
        exemplar_texts = []

    augmented: List[Example] = []
    choices_pool = outcome.synonyms or exemplar_texts or [outcome.label.lower()]
    for _ in range(augment_factor):
        if templates:
            template = rng.choice(templates)
        else:
            template = lambda word: f"Please take care of {word}."
        token = rng.choice(choices_pool)
        text = template(token)
        # Add minor variation with trailing instruction occasionally
        if rng.random() < 0.3:
            text = text.rstrip(".") + " as soon as possible."
        augmented.append(Example(text=text, label=outcome.outcome_id))
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a simple dataset for PyTorch training.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Path to the config directory containing outcomes/ (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_training") / "data",
        help="Directory where train/val splits will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of examples assigned to the training split (default: %(default)s)",
    )
    parser.add_argument(
        "--min-val-per-label",
        type=int,
        default=DEFAULT_MIN_VAL_PER_LABEL,
        help="Minimum validation examples to keep per label when possible (default: %(default)s)",
    )
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=DEFAULT_AUGMENT_FACTOR,
        help="Synthetic examples to generate per outcome (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling (default: %(default)s)",
    )
    args = parser.parse_args()

    outcome_data = read_outcomes(args.config_dir)
    examples: List[Example] = []
    for outcome in outcome_data:
        examples.extend(outcome.examples)
        examples.extend(generate_augmented_examples(outcome, args.augment_factor, args.seed))

    vocab = build_vocabulary(examples)
    train_split, val_split = stratified_split(
        examples,
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_val_per_label=args.min_val_per_label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vocab(args.output_dir / "simple_vocab.json", vocab)
    write_jsonl(args.output_dir / "train.jsonl", train_split)
    write_jsonl(args.output_dir / "val.jsonl", val_split)

    summary = {
        "total_examples": len(examples),
        "train_examples": len(train_split),
        "validation_examples": len(val_split),
        "vocab_size": len(vocab),
        "augment_factor": args.augment_factor,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

