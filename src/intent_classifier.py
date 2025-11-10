from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class Prediction:
    outcome_id: Optional[str]
    confidence: float
    is_unknown: bool


class SingleLabelModel:
    """Fallback model when only a single outcome has examples."""

    def __init__(self, outcome_id: str):
        self.outcome_id = outcome_id

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        return [[1.0] for _ in texts]

    def predict(self, texts: List[str]) -> List[str]:
        return [self.outcome_id for _ in texts]


class IntentClassifier:
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.settings_path = self.config_dir / "settings.json"
        self.outcomes_dir = self.config_dir / "outcomes"
        self.archive_dir = self.config_dir / "outcomes_old"
        self.legacy_config_path = self.config_dir / "outcomes.json"

        self._pipeline: Optional[Pipeline] = None
        self._classes: List[str] = []
        self._label_index: Dict[str, int] = {}
        self._outcome_files: Dict[str, Path] = {}

        self._ensure_structure()
        self._maybe_migrate_legacy_config()

        self.config = self._load_config()
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def _ensure_structure(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_path.exists():
            self._persist_settings({"teach_mode": {"unknown_threshold": 0.5}})

    def _maybe_migrate_legacy_config(self) -> None:
        if not self.legacy_config_path.exists():
            return
        try:
            with self.legacy_config_path.open("r", encoding="utf-8") as handle:
                legacy_config = json.load(handle)
        except json.JSONDecodeError:
            return

        for outcome in legacy_config.get("outcomes", []):
            outcome_id = outcome.get("id")
            if not outcome_id:
                continue
            destination = self.outcomes_dir / f"{outcome_id}.json"
            if destination.exists():
                continue
            self._write_outcome_file(destination, outcome)

        teach_mode = legacy_config.get("teach_mode")
        if teach_mode:
            self._persist_settings({"teach_mode": teach_mode})

        backup_path = self.legacy_config_path.with_suffix(".json.bak")
        try:
            self.legacy_config_path.rename(backup_path)
        except OSError:
            pass

    def _persist_settings(self, data: Dict) -> None:
        with self.settings_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _load_config(self) -> Dict:
        teach_mode = {"unknown_threshold": 0.5}
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                settings = json.load(handle)
                teach_mode = settings.get("teach_mode", teach_mode)
        except (FileNotFoundError, json.JSONDecodeError):
            self._persist_settings({"teach_mode": teach_mode})

        outcomes: List[Dict] = []
        self._outcome_files.clear()
        for outcome_path in sorted(self.outcomes_dir.glob("*.json")):
            try:
                with outcome_path.open("r", encoding="utf-8") as handle:
                    outcome = json.load(handle)
            except json.JSONDecodeError:
                continue
            outcome_id = outcome.get("id")
            if not outcome_id:
                continue
            outcomes.append(outcome)
            self._outcome_files[outcome_id] = outcome_path

        return {"teach_mode": teach_mode, "outcomes": outcomes}

    def _write_outcome_file(self, path: Path, outcome: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(outcome, handle, indent=2, ensure_ascii=False)

    def _build_normalization_maps(self) -> Tuple[Dict[str, str], List[str]]:
        synonym_map: Dict[str, str] = {}
        vocabulary: List[str] = []
        for outcome in self.config.get("outcomes", []):
            canonical = outcome["synonyms"][0] if outcome.get("synonyms") else outcome["label"]
            for bucket in ("synonyms", "spell_variants"):
                for term in outcome.get(bucket, []) or []:
                    lower = term.lower()
                    synonym_map[lower] = canonical.lower()
                    vocabulary.append(lower)
            vocabulary.extend(self._tokenize(outcome["label"]))
            for example in outcome.get("examples", []):
                vocabulary.extend(self._tokenize(example["text"]))
        # Deduplicate vocabulary while preserving order
        seen = set()
        deduped: List[str] = []
        for token in vocabulary:
            if token not in seen:
                seen.add(token)
                deduped.append(token)
        return synonym_map, deduped

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _lemmatize_token(self, token: str) -> str:
        # Simple heuristic lemmatization to keep dependencies light.
        if len(token) <= 3:
            return token
        for suffix in ("ing", "ies", "ied", "s", "ed"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                if suffix == "ies":
                    return token[:-3] + "y"
                if suffix == "ied":
                    return token[:-3] + "y"
                return token[: -len(suffix)]
        return token

    def _normalize_text(self, text: str) -> str:
        tokens = self._tokenize(text)
        normalized: List[str] = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self._synonym_map:
                normalized.append(self._synonym_map[token_lower])
                continue
            match = get_close_matches(token_lower, self._vocabulary, n=1, cutoff=0.85)
            if match:
                normalized.append(match[0])
                continue
            normalized.append(self._lemmatize_token(token_lower))
        return " ".join(normalized)

    def _build_dataset(self) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        for outcome in self.config.get("outcomes", []):
            outcome_id = outcome["id"]
            for example in outcome.get("examples", []):
                text = example["text"]
                normalized = self._normalize_text(text)
                texts.append(normalized)
                labels.append(outcome_id)
        return texts, labels

    def retrain(self) -> None:
        texts, labels = self._build_dataset()
        if not texts:
            self._pipeline = None
            self._label_index = {}
            return

        unique_labels = sorted(set(labels))
        self._label_index = {label: idx for idx, label in enumerate(unique_labels)}

        self._classes = unique_labels

        if len(unique_labels) == 1:
            self._pipeline = SingleLabelModel(unique_labels[0])  # type: ignore[assignment]
            return

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        classifier = LogisticRegression(max_iter=1000)
        self._pipeline = Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", classifier),
            ]
        )
        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.named_steps["clf"].classes_)  # type: ignore[index]

    def predict(self, text: str) -> Prediction:
        if not text.strip():
            return Prediction(outcome_id=None, confidence=0.0, is_unknown=True)

        normalized = self._normalize_text(text)
        if not self._pipeline:
            return Prediction(outcome_id=None, confidence=0.0, is_unknown=True)

        teach_mode = self.config.get("teach_mode", {})
        threshold = float(teach_mode.get("unknown_threshold", 0.5))

        if isinstance(self._pipeline, SingleLabelModel):
            outcome_id = self._pipeline.outcome_id
            confidence = 1.0
        else:
            proba = self._pipeline.predict_proba([normalized])[0]
            max_index = int(proba.argmax())
            confidence = float(proba[max_index])
            outcome_id = self._classes[max_index]

        is_unknown = confidence < threshold
        return Prediction(outcome_id=outcome_id if not is_unknown else None, confidence=confidence, is_unknown=is_unknown)

    def reload(self) -> None:
        self.config = self._load_config()
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def add_outcome(self, outcome: Dict) -> None:
        outcome_id = outcome.get("id")
        if not outcome_id:
            raise ValueError("Outcome must include an 'id'.")
        path = self.outcomes_dir / f"{outcome_id}.json"
        if path.exists():
            raise ValueError(f"Outcome '{outcome_id}' already exists.")
        self._write_outcome_file(path, outcome)
        self.reload()

    def add_example(self, outcome_id: str, example: Dict) -> None:
        path = self._outcome_files.get(outcome_id)
        if not path:
            raise ValueError(f"Unknown outcome id: {outcome_id}")
        with path.open("r", encoding="utf-8") as handle:
            outcome = json.load(handle)
        outcome.setdefault("examples", []).append(example)
        self._write_outcome_file(path, outcome)
        self.reload()

    def update_synonyms(
        self,
        outcome_id: str,
        synonyms: Optional[List[str]] = None,
        spell_variants: Optional[List[str]] = None,
    ) -> None:
        path = self._outcome_files.get(outcome_id)
        if not path:
            raise ValueError(f"Unknown outcome id: {outcome_id}")
        with path.open("r", encoding="utf-8") as handle:
            outcome = json.load(handle)
        if synonyms is not None:
            outcome["synonyms"] = sorted(set(s.lower() for s in synonyms))
        if spell_variants is not None:
            outcome["spell_variants"] = sorted(set(s.lower() for s in spell_variants))
        self._write_outcome_file(path, outcome)
        self.reload()

    def remove_outcome(self, outcome_id: str) -> None:
        path = self._outcome_files.get(outcome_id)
        if not path or not path.exists():
            raise ValueError(f"Unknown outcome id: {outcome_id}")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"{outcome_id}_{timestamp}.json"
        try:
            shutil.move(str(path), archive_path)
        except OSError:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.reload()

    def reset_outcomes(self) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        destination = self.archive_dir / timestamp
        destination.mkdir(parents=True, exist_ok=True)
        for outcome_path in self.outcomes_dir.glob("*.json"):
            try:
                shutil.move(str(outcome_path), destination / outcome_path.name)
            except OSError:
                pass
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        self.reload()


def load_classifier(config_dir: str | Path) -> IntentClassifier:
    return IntentClassifier(Path(config_dir))

