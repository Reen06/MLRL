# Model Training Playbook (Forward Plan)

This document captures the end-to-end recipe for retraining and maintaining the PyTorch intent classifier going forward.

---

## 1. When to Retrain
- Outcome definitions (`config/outcomes/*.json`) change materially (new outcomes, lots of new examples, synonym updates).
- The simple classifier’s validation accuracy drops below your acceptable threshold.
- Prior to a major release or deployment to new use cases.

Aim for at least a monthly refresh if outcomes evolve rapidly.

---

## 2. Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r model_training/requirements-train.txt
```
> `requirements-train.txt` now includes NumPy so PyTorch utilities run cleanly.

---

## 3. Build a Large Training Set
Regenerate the dataset with **heavy synthetic augmentation** so every outcome is represented by hundreds of paraphrases. This step is where you decide how “huge” the dataset should be.

```bash
python model_training/scripts/prepare_simple_dataset.py \
  --augment-factor 500 \        # 500 synthetic sentences per outcome
  --train-fraction 0.90 \       # keep 10% for validation
  --min-val-per-label 10 \      # guarantee at least 10 val samples/label
  --seed 2025                   # set for reproducibility
```

This will expand a small curated set (e.g., 18 originals) into ~9,000 training rows plus a sizable validation set. Tweak `--augment-factor` up/down depending on hardware and patience:
- 100–200: minutes of training
- 500+: several minutes (CPU) or faster on GPU
- 1000+: can take 10+ minutes on CPU; expect higher memory usage

---

## 4. Run a Deep Training Loop
Use a wider/deeper network with dropout and weight decay to prevent overfitting. Target **1000 epochs** for a multi-minute run:

```bash
python model_training/scripts/train_simple_model.py \
  --epochs 1000 \
  --lr 0.003 \
  --batch-size 64 \
  --hidden-size 512 \
  --hidden-layers 4 \
  --dropout 0.45 \
  --weight-decay 1e-4 \
  --checkpoint model_training/models/simple_model.pt
```

Tips:
- **GPU**: set `CUDA_VISIBLE_DEVICES=...` or run inside a CUDA-capable container for major speedups.
- **Long runs**: consider reducing `--augment-factor` slightly or lowering epochs if training exceeds available time.
- Training logs already show loss/accuracy each epoch. Expect validation accuracy to exceed 95% before epoch 200 if augmentation is working.

---

## 5. Evaluate & Sanity-Check
```bash
python model_training/scripts/evaluate_simple_model.py \
  --checkpoint model_training/models/simple_model.pt
```
- Confirms the checkpoint matches the saved architecture (the script reads the `config` payload embedded in the checkpoint).
- Prints validation accuracy and number of evaluation samples.

Optional sanity tests (run manually):
```python
from model_training.simple_inference import load_classifier, predict
model, vocab, idx_to_label = load_classifier("model_training/models/simple_model.pt")
examples = [
    "Remind me to pay the electric bill tomorrow",
    "Schedule a tune-up for the car next week",
    "Book a flight to Denver in March"
]
for text in examples:
    label, confidence = predict(text, model, vocab, idx_to_label)
    print(text, "->", label, f"{confidence:.2%}")
```

---

## 6. Deploy the Updated Model
1. Store the checkpoint + metrics (`simple_model.pt`, `simple_model.pt.metrics.json`) with a versioned name, e.g., `simple_model_v2025-11.pt`.
2. Update your CLI or inference helper to load the new file path.
3. Add release notes if behavior changes (e.g., new labels, better coverage).

For production, keep the last known-good checkpoint around so you can roll back quickly.

---

## 7. Troubleshooting & Scaling
- **Memory Errors**: Lower `--augment-factor` or `--batch-size`, or train on GPU.
- **Slow Training**: Use GPU (NVIDIA 3070 Ti is plenty), or drop epochs to 500 with slightly higher `--lr`.
- **Validation Accuracy Drops**: Inspect `model_training/data/val.jsonl` to ensure each outcome has enough representation; adjust `--min-val-per-label`.
- **Overfitting (train ≫ val)**: Increase dropout/weight decay, or enrich synthetic paraphrases (`--augment-factor` up; synonyms/teach hints in outcome files).

---

## 8. Future Enhancements (Ideas)
- Replace bag-of-words with embeddings (GloVe, BERT) for semantic understanding.
- Incorporate teach-mode feedback incrementally rather than full retrains.
- Automate the pipeline with a Makefile or CI job to regenerate and benchmark models nightly.
- Export to ONNX for faster inference in production environments.

---

**Next Scheduled Action:** Re-run the full pipeline after your next batch of outcome updates or before the next major release. Keep this document close—it’s your training playbook.

