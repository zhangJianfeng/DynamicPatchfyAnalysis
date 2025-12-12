import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        # input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        seq_len: int = 64,
        mlp_hidden: int = 32,
        dropout: float = 0.0,
        patch: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(patch, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len//patch, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: MLP that maps the flattened encoded representation back to the sequence
        self.decoder = nn.Sequential(
            nn.Linear((seq_len//patch) * d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, seq_len),
        )
        self.patch = patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        batch = x.size(0)
        x = x.reshape(batch, -1, self.patch)  # (batch, seq_len // patch, patch)
        x = self.input_proj(x)  # (batch, seq_len // patch, d_model)
        x = self.pos_enc(x)
        enc = self.encoder(x)  # (batch, seq_len // patch, d_model)
        flat = enc.reshape(batch, -1)
        out = self.decoder(flat)  # (batch, seq_len)
        out = out.unsqueeze(-1)  # (batch, seq_len, 1)
        return out


def build_model(**kwargs) -> TransformerAutoencoder:
    return TransformerAutoencoder(**kwargs)
