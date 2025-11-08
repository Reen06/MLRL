# NLP Outcome Router

Lightweight Python CLI for routing natural language inputs to configurable outcomes. A small TF-IDF + logistic regression model interprets user phrasing (including synonyms and minor typos) and maps it to outcomes stored in a JSON file. Unknown phrases trigger an interactive teach mode so the system improves over time.

## Features

- CLI menu for classifying text, managing outcomes, examples, and synonym lists
- Config-driven design with all outcomes stored in `config/outcomes.json`
- Normalization pipeline performs lowercasing, simple lemmatization, synonym expansion, and fuzzy spell correction
- Teach mode captures unknown requests and lets you assign or create outcomes on the fly
- Lightweight dependency footprint (`scikit-learn`) and no external services

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install scikit-learn
python src/cli.py
```

The first run creates `config/outcomes.json` if it is missing. A sample “food expiration” outcome is included for reference.

## Workflow

1. **Classify text** – enter natural language (e.g., “My yogurt goes bad next Monday”). If confidence is high, the matching outcome is shown.
2. **Teach mode** – when the classifier is unsure, you can assign the text to an existing outcome or create a brand-new outcome. The labeled example is saved and the model retrains immediately.
3. **Manage outcomes** – use the menu to review outcomes, add curated examples, or adjust synonym/spelling lists for better coverage.

## Config Structure

`config/outcomes.json` contains:

- `outcomes`: array of outcome objects with:
  - `id`, `label`, `action` metadata for downstream scripts
  - `synonyms` and `spell_variants` to guide normalization
  - `examples` (text plus optional key/value labels)
  - `teach_hint` shown during teach mode
- `teach_mode`: thresholds and prompts for unknown handling

Copy this file to share the learned outcomes with other deployments.

## Example

Sample entry in the default config:

```startLine:endLine:config/outcomes.json
    {
      "id": "food_expiration",
      "label": "Food Expiration Tracking",
      "synonyms": [
        "expiration",
        "expires",
        "expiry",
        "goes bad",
        "spoils"
      ],
      "spell_variants": [
        "expries",
        "expiraton"
      ],
      "action": {
        "type": "food_expiration",
        "fields": [
          "item",
          "expires_on"
        ]
      ],
      "examples": [
        {
          "text": "I bought a new jug of milk that expires in two weeks.",
          "labels": {
            "item": "milk",
            "expires_in": "two weeks"
          }
        },
        {
          "text": "These eggs go bad on Friday.",
          "labels": {
            "item": "eggs",
            "expires_on": "Friday"
          }
        }
      ],
      "teach_hint": "Collect expiration info for groceries."
    }
```

Extend the config with more outcomes (e.g., bill reminders, maintenance tasks) and add examples using the CLI to improve accuracy.

