from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._pipeline: Optional[Pipeline] = None
        self._classes: List[str] = []
        self._label_index: Dict[str, int] = {}
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

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
        deduped = []
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

    def add_outcome(self, outcome: Dict) -> None:
        self.config.setdefault("outcomes", []).append(outcome)
        self._persist_config()
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def add_example(self, outcome_id: str, example: Dict) -> None:
        for outcome in self.config.get("outcomes", []):
            if outcome["id"] == outcome_id:
                outcome.setdefault("examples", []).append(example)
                break
        else:
            raise ValueError(f"Unknown outcome id: {outcome_id}")
        self._persist_config()
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def update_synonyms(self, outcome_id: str, synonyms: Optional[List[str]] = None, spell_variants: Optional[List[str]] = None) -> None:
        for outcome in self.config.get("outcomes", []):
            if outcome["id"] == outcome_id:
                if synonyms is not None:
                    outcome["synonyms"] = sorted(set(s.lower() for s in synonyms))
                if spell_variants is not None:
                    outcome["spell_variants"] = sorted(set(s.lower() for s in spell_variants))
                break
        else:
            raise ValueError(f"Unknown outcome id: {outcome_id}")
        self._persist_config()
        self._synonym_map, self._vocabulary = self._build_normalization_maps()
        self.retrain()

    def _persist_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)


def load_classifier(config_path: str | Path) -> IntentClassifier:
    return IntentClassifier(Path(config_path))

