# Univariate Time Series Autoencoder

This repo contains a simple PyTorch implementation of a univariate time-series autoencoder where:
- Encoder: Transformer-based encoder
- Decoder: MLP that reconstructs the input sequence

Quick start:

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train with synthetic data:

```bash
python -m src.train --epochs 20 --seq-len 64 --batch-size 64
```

Files:
- `src/model.py`: Transformer encoder + MLP decoder
- `src/dataset.py`: synthetic sine dataset and loader
- `src/train.py`: training loop and checkpointing

Adjust hyperparameters via CLI flags in `src/train.py`.