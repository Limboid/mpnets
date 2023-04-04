from exports import export
import torch


@export
def sparse_threshold(self, Z, threshold=1):
    return torch.sign(Z) * torch.relu(torch.abs(Z) - threshold)


@export
def soft_v(self, x):
    return x * torch.tanh(x) + 1
