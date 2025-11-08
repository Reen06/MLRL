from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from intent_classifier import IntentClassifier, Prediction


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "outcomes.json"


def prompt(message: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    response = input(f"{message}{suffix}: ").strip()
    if not response and default is not None:
        return default
    return response


def pause() -> None:
    input("\nPress Enter to continue...")


def list_outcomes(classifier: IntentClassifier) -> None:
    print("\nConfigured Outcomes:\n")
    for outcome in classifier.config.get("outcomes", []):
        print(f"- {outcome['id']} :: {outcome['label']}")
        if outcome.get("synonyms"):
            print(f"  Synonyms: {', '.join(outcome['synonyms'])}")
        if outcome.get("spell_variants"):
            print(f"  Spell Variants: {', '.join(outcome['spell_variants'])}")
        example_count = len(outcome.get("examples", []))
        print(f"  Training examples: {example_count}")
    if not classifier.config.get("outcomes"):
        print("  (none yet)")


def choose_outcome(classifier: IntentClassifier) -> Optional[str]:
    outcomes = classifier.config.get("outcomes", [])
    if not outcomes:
        print("No outcomes configured yet.")
        return None
    print("\nSelect an outcome:")
    for idx, outcome in enumerate(outcomes, start=1):
        print(f"{idx}. {outcome['label']} ({outcome['id']})")
    choice = prompt("Enter number or leave blank to cancel", "")
    if not choice:
        return None
    try:
        index = int(choice) - 1
    except ValueError:
        print("Invalid selection.")
        return None
    if index < 0 or index >= len(outcomes):
        print("Selection out of range.")
        return None
    return outcomes[index]["id"]


def collect_labels() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    print("\nAdd key/value labels for this example (optional). Leave key blank to finish.")
    while True:
        key = prompt("Label key", "")
        if not key:
            break
        value = prompt(f"Value for '{key}'")
        labels[key] = value
    return labels


def add_outcome_flow(classifier: IntentClassifier) -> Optional[str]:
    label = prompt("Outcome label (human readable)")
    outcome_id = prompt("Outcome id (no spaces)", label.lower().replace(" ", "_"))
    synonyms_raw = prompt("Comma separated synonyms (optional)", "")
    spell_raw = prompt("Common misspellings/variants (optional)", "")
    teach_hint = prompt("Teach hint (shown during learning)", f"Handle {label.lower()}")

    outcome = {
        "id": outcome_id,
        "label": label,
        "synonyms": [s.strip().lower() for s in synonyms_raw.split(",") if s.strip()],
        "spell_variants": [s.strip().lower() for s in spell_raw.split(",") if s.strip()],
        "examples": [],
        "teach_hint": teach_hint,
        "action": {
            "type": outcome_id,
            "fields": []
        },
    }
    classifier.add_outcome(outcome)
    print(f"Added outcome '{outcome_id}'.")
    return outcome_id


def add_example_flow(classifier: IntentClassifier, outcome_id: Optional[str] = None, text: Optional[str] = None) -> None:
    target_id = outcome_id or choose_outcome(classifier)
    if not target_id:
        return
    example_text = text or prompt("Example text")
    labels = collect_labels()
    classifier.add_example(target_id, {"text": example_text, "labels": labels})
    print(f"Added example to '{target_id}'.")


def update_synonyms_flow(classifier: IntentClassifier) -> None:
    outcome_id = choose_outcome(classifier)
    if not outcome_id:
        return
    synonyms_raw = prompt("New comma separated synonyms (leave blank to keep current)", "")
    spell_raw = prompt("New comma separated spell variants (leave blank to keep current)", "")
    synonyms = None if not synonyms_raw else [s.strip().lower() for s in synonyms_raw.split(",") if s.strip()]
    spell_variants = None if not spell_raw else [s.strip().lower() for s in spell_raw.split(",") if s.strip()]
    classifier.update_synonyms(outcome_id, synonyms=synonyms, spell_variants=spell_variants)
    print(f"Updated synonyms for '{outcome_id}'.")


def teach_mode(classifier: IntentClassifier, text: str) -> None:
    teach_mode_config = classifier.config.get("teach_mode", {})
    prompt_text = teach_mode_config.get(
        "confirmation_prompt",
        "I didn't recognize that input. How should I handle it?"
    )
    print(f"\nTeach Mode: {prompt_text}")
    print("1. Assign to existing outcome")
    print("2. Create a new outcome")
    choice = prompt("Select an option", "1")
    if choice == "1":
        target_id = choose_outcome(classifier)
        if target_id:
            add_example_flow(classifier, outcome_id=target_id, text=text)
    elif choice == "2":
        new_outcome_id = add_outcome_flow(classifier)
        if new_outcome_id:
            add_example_flow(classifier, outcome_id=new_outcome_id, text=text)
    else:
        print("Teach mode cancelled.")


def classify_flow(classifier: IntentClassifier) -> None:
    text = prompt("Enter input text")
    prediction: Prediction = classifier.predict(text)
    if prediction.is_unknown or not prediction.outcome_id:
        print("\nNo confident match found.")
        teach_mode(classifier, text)
    else:
        print("\nPrediction:")
        print(f"- Outcome: {prediction.outcome_id}")
        print(f"- Confidence: {prediction.confidence:.2f}")


def ensure_config_exists(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "outcomes": [], "teach_mode": {"unknown_threshold": 0.5}}, handle, indent=2)


def main() -> None:
    ensure_config_exists(CONFIG_PATH)
    classifier = IntentClassifier(CONFIG_PATH)

    actions = {
        "1": ("Classify text", classify_flow),
        "2": ("Add outcome", add_outcome_flow),
        "3": ("Add example", add_example_flow),
        "4": ("Update synonyms", update_synonyms_flow),
        "5": ("List outcomes", list_outcomes),
        "6": ("Quit", None),
    }

    while True:
        print("\n=== NLP Outcome Router ===")
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        choice = prompt("Select option", "1")
        if choice == "6":
            print("Goodbye!")
            break
        action = actions.get(choice)
        if not action:
            print("Invalid selection.")
            continue
        label, func = action
        if func is None:
            break
        try:
            func(classifier)  # type: ignore[arg-type]
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error during '{label}': {exc}")
        pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

