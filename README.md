# NLP Outcome Router

Lightweight Python CLI for routing natural language inputs to configurable outcomes. A small TF-IDF + logistic regression model interprets user phrasing (including synonyms and minor typos) and maps it to outcomes stored in a JSON file. Unknown phrases trigger an interactive teach mode so the system improves over time.

## Features

- CLI menu for classifying text, managing outcomes, examples, and synonym lists
- Config-driven design with outcomes stored as individual files under `config/outcomes/`
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

The first run ensures `config/settings.json` and an `config/outcomes/` directory exist. A sample “food expiration” outcome file is included for reference.

## Docker

Run these commands from the project root (`MLRL` directory):

- Development container (mounts your local code into the container so changes reflect immediately):

  ```bash
  docker build -f Dockerfile.dev -t nlp-router-dev .
  docker run -it --rm -v ${PWD}:/app nlp-router-dev
  ```

- Shareable image (packages the entire project into the image):

  ```bash
  docker build -t nlp-router .
  docker run -it --rm nlp-router
  ```

## Model Training

For an end-to-end example of training a lightweight classifier in plain PyTorch (no Hugging Face dependencies), check `model_training/`. The scripts there:

- turn the outcome examples into a minimal dataset (`prepare_simple_dataset.py`)
- train a small feed-forward network (`train_simple_model.py`)
- evaluate saved checkpoints (`evaluate_simple_model.py`)

See `model_training/README.md` for the 3-command quick start.
For a production-scale checklist—including large synthetic dataset generation and 1000-epoch training—refer to `GOING_FORWARD.md`.

## Workflow

1. **Classify text** – enter natural language (e.g., “My yogurt goes bad next Monday”). If confidence is high, the matching outcome is shown.
2. **Teach mode** – when the classifier is unsure, you can assign the text to an existing outcome or create a brand-new outcome. The labeled example is saved and the model retrains immediately.
3. **Manage outcomes** – use the menu to review outcomes, add curated examples, or adjust synonym/spelling lists for better coverage.

## Config Structure

Configuration lives in the `config/` directory:

- `config/outcomes/` – one JSON file per outcome. Each file includes:
  - `id`, `label`, `action` metadata for downstream scripts
  - `synonyms` and `spell_variants` to guide normalization
  - `examples` (text plus optional key/value labels)
  - `teach_hint` shown during teach mode
- `config/settings.json` – global settings such as `teach_mode` thresholds and prompts.

Copy the outcomes directory (and settings, if desired) to share the learned outcomes with other deployments.

## Example

Sample outcome file (`config/outcomes/food_expiration.json`):

```1:33:config/outcomes/food_expiration.json
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

