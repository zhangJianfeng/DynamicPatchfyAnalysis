import torch


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, device='cpu') -> dict:
    return torch.load(path, map_location=device)
