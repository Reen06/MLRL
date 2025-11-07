# Simple MNIST Classifier

A beginner-friendly PyTorch project that trains and evaluates a convolutional
neural network on the MNIST handwritten digit dataset. Everything runs on CPU,
and the dataset is downloaded automatically the first time you launch training
or evaluation.

## Features 
 
- Minimal dependency list (`torch`, `torchvision`, `numpy`)
- Automatic MNIST download into `data/`
- Small CNN that reaches >97% accuracy in a few minutes on CPU
- Separate scripts for training (`src/train.py`) and evaluation (`src/test.py`)
- Clear, inline comments that explain the most important steps

## Prerequisites

- Python 3.9+ (3.11 works great)
- `pip` for managing packages

Install the required libraries once:

```bash
pip install -r requirements.txt
```

## How to Train

Run the training module to download MNIST (if needed) and fit the model:

```bash
python -m src.train
```

By default this:

- trains for 3 epochs with a batch size of 64,
- saves the checkpoint to `data/mnist_cnn.pt`,
- prints training and test accuracy after every epoch.

You can tweak hyperparameters from the command line, e.g.:

```bash
python -m src.train --epochs 5 --batch-size 128 --learning-rate 5e-4
```

Additional options:

- `--data-dir`: custom dataset/cache directory (default: `data/`)
- `--checkpoint-path`: where to save the trained weights (default: `data/mnist_cnn.pt`)
- `--seed`: control the random seed

## How to Evaluate

Once you have a checkpoint, run:

```bash
python -m src.test
```

This loads the saved weights and reports test-set accuracy. Useful flags:

- `--data-dir`: must match the directory used during training (defaults to `data/`)
- `--checkpoint-path`: path to the `.pt` file you want to evaluate
- `--batch-size`: controls evaluation batch size (256 by default for faster inference)

If the checkpoint is missing, the script reminds you to train the model first.

## Project Layout

```
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
├── data/                    # Auto-created; stores MNIST & checkpoints
├── src/
│   ├── model.py             # CNN definition
│   ├── train.py             # Training loop + CLI
│   └── test.py              # Evaluation script + CLI
```

## Next Steps

- Try experimenting with the network architecture in `src/model.py`
- Adjust training epochs or learning rate to push accuracy higher
- Extend `src/test.py` to produce confusion matrices or save predictions
