import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from src.model import build_model
from src.dataset import SyntheticSineDataset, collate_fn
from src.utils import save_checkpoint


import pandas as pd
import numpy as np
import random

ETTH = pd.read_csv('./ETTh1.csv').OT.to_numpy()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--patch', type=int, default=1)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--log-dir', type=str, default=None, help='TensorBoard log directory (default: <save-dir>/logs)')
    # p.add_argument('--num-series', type=int, default=5000)
    return p.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # tensorboard writer (optional)
    log_dir = args.log_dir or os.path.join(args.save_dir, 'logs')
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
        print('Warning: tensorboard not installed; proceeding without SummaryWriter')

    dataset = []
    for i in range(0, len(ETTH) - args.seq_len, 20):
        seq = ETTH[i:i + args.seq_len]
        dataset.append(seq)
    dataset = torch.tensor(np.array(dataset), dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(seq_len=args.seq_len, patch=args.patch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for xb in loader:
            xb = xb.to(device)
            preds = model(xb)
            loss = criterion(preds, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.6f} - time: {time.time()-t0:.1f}s")

        # log epoch loss
        if writer is not None:
            writer.add_scalar('train/loss', epoch_loss, epoch)

        # save checkpoint per epoch and keep best
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
        }
        save_checkpoint(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pt'))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt, os.path.join(args.save_dir, 'best.pt'))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    train()
