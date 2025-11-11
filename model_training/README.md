# Simple PyTorch Training Workflow

This directory now hosts a lightweight reference pipeline that trains a small PyTorch model on the outcome examples already checked into the repository. No Hugging Face dependencies, quantization tricks, or external model downloads are required—everything runs on CPU or a single GPU with standard PyTorch.

## Layout

- `scripts/`
  - `prepare_simple_dataset.py` – converts the texts in `config/outcomes/*.json` into a minimal dataset (train/val splits plus vocabulary metadata).
  - `train_simple_model.py` – trains a bag-of-words classifier using plain `torch.nn` modules.
  - `evaluate_simple_model.py` – loads a saved checkpoint and reports validation accuracy.
- `data/` – generated artifacts:
  - `simple_vocab.json`
  - `train.jsonl`, `val.jsonl`
- `models/` – trained checkpoints (`state_dict.pt` plus metrics).

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r model_training/requirements-train.txt

# Generate dataset from config/outcomes, with heavy synthetic augmentation
python model_training/scripts/prepare_simple_dataset.py \
  --augment-factor 500 \
  --train-fraction 0.90 \
  --min-val-per-label 10

# Train the classifier (≈10 minutes on GPU / tens of minutes on CPU)
python model_training/scripts/train_simple_model.py \
  --epochs 1000 \
  --lr 0.003 \
  --batch-size 64 \
  --hidden-size 512 \
  --hidden-layers 4 \
  --dropout 0.45 \
  --weight-decay 1e-4 \
  --checkpoint model_training/models/simple_model.pt

# Evaluate
python model_training/scripts/evaluate_simple_model.py \
  --checkpoint model_training/models/simple_model.pt
```

The training script prints loss/accuracy per epoch, and saves the best-performing weights along with a JSON metrics report (`models/simple_model_metrics.json`).

## Notes

- The dataset is tiny (derived solely from the curated examples), so expect modest accuracy unless you add more examples to `config/outcomes/`.
- Vocabulary building is case-insensitive and strips punctuation; feel free to tweak the tokenizer inside `prepare_simple_dataset.py` if you need something more sophisticated.
- Increase `--augment-factor`, hidden size, or number of epochs if you want longer training runs or higher accuracy. The settings above generate ~9k synthetic examples and train for ~1000 epochs—plan on 10 minutes (GPU) to 30+ minutes (CPU).
- Because everything is in vanilla PyTorch, you can easily extend the training loop (e.g., add embedding layers, experiment with CNN/RNN architectures, or integrate data augmentation) without worrying about external model formats.

