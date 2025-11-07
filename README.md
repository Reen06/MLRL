# Simple MNIST Classifier

This project demonstrates a lightweight PyTorch workflow for training and evaluating a handwritten digit classifier on the MNIST dataset. The code automatically downloads the dataset, trains a small neural network on CPU, and provides a basic evaluation script.

## Requirements

- Python 3.9 or newer is recommended.
- Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

This command will download MNIST (if it is not already present in `data/`), build the model, and run a brief training session on CPU.

## Testing

```bash
python -m src.test
```

The testing script loads the saved weights from training and reports basic accuracy metrics.

## Project Structure

```
├── requirements.txt
├── README.md
├── data/                # Auto-created and populated by the scripts
├── src/
│   ├── model.py
│   ├── train.py
│   └── test.py
```

Each source file is extensively commented to highlight the most important steps for beginners.
