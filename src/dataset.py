import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset







class SyntheticSineDataset(Dataset):
    """Generate random sine waves with noise for univariate time series reconstruction."""

    def __init__(self, num_series: int = 10000, seq_len: int = 64, freq_range=(0.1, 1.0), noise_std=0.05):
        super().__init__()
        self.num_series = num_series
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.noise_std = noise_std

    def __len__(self):
        return self.num_series

    def __getitem__(self, idx):
        freq = random.uniform(*self.freq_range)
        phase = random.uniform(0, 2 * math.pi)
        t = np.linspace(0, 2 * math.pi, self.seq_len)
        series = np.sin(freq * t + phase)
        series = series + np.random.normal(scale=self.noise_std, size=self.seq_len)
        series = series.astype(np.float32)
        series = series.reshape(self.seq_len, 1)
        return torch.from_numpy(series)


def collate_fn(batch):
    # batch is list of tensors (seq_len, 1)
    x = torch.stack(batch, dim=0)  # (batch, seq_len, 1)
    return x
