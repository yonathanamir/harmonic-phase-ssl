"""The network of Itoyama et al. (2021) with a swappable phase embedding.

    z = [g(x_1); ...; g(x_J)]          J = 8 microphones x 256 bins, D features per value
    u = W1 z + b1                      H = 256 hidden units
    h = BN(ReLU(u))
    p = softmax(W2 h + b2)             72 direction classes of 5 degrees

Only the embedding g changes between arms. In training mode the phases get the per-bin
common-phase augmentation: one random phase psi_k per sample and bin, added to all channels.
It keeps every inter-channel phase difference and stops the first layer from memorizing the
absolute phases of the training set.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from embeddings import make_embedding


def common_phase_shift(phase: torch.Tensor) -> torch.Tensor:
    """Theta_mk <- Theta_mk + psi_k (mod 2 pi), psi_k ~ U[0, 2 pi) per sample and bin."""
    psi = 2 * torch.pi * torch.rand(phase.shape[0], 1, phase.shape[2],
                                    device=phase.device, dtype=phase.dtype)
    return torch.remainder(phase + psi + torch.pi, 2 * torch.pi) - torch.pi


class SSLNet(nn.Module):
    """embedding -> Linear -> ReLU -> BatchNorm -> Linear (softmax is in the loss).

    The first projection runs in chunks of `projection_chunk` samples with activation
    checkpointing, so that a batch of 8192 fits on a 24 GB GPU; BatchNorm still sees the
    complete batch at once.
    """

    def __init__(self, embedding: str, n_filters: int = 2, n_inputs: int = 2048,
                 hidden: int = 256, n_classes: int = 72, projection_chunk: int = 512):
        super().__init__()
        self.embedding = make_embedding(embedding, n_inputs, n_filters)
        self.fc1 = nn.Linear(n_inputs * self.embedding.d, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.projection_chunk = projection_chunk

    @property
    def needs_mag(self) -> bool:
        return self.embedding.needs_mag

    def project(self, phase, mag):
        z = self.embedding(phase, mag) if self.needs_mag else self.embedding(phase)
        return self.fc1(z)

    def forward(self, phase, mag=None):                               # phase: (B, 8, 256)
        if self.training:
            phase = common_phase_shift(phase)
        phase = phase.flatten(1)
        mag = None if mag is None else mag.flatten(1)
        chunk = self.projection_chunk or len(phase)
        parts = []
        for lo in range(0, len(phase), chunk):
            x = phase[lo:lo + chunk]
            a = None if mag is None else mag[lo:lo + chunk]
            if self.training and torch.is_grad_enabled() and self.projection_chunk:
                parts.append(checkpoint(self.project, x, a, use_reentrant=False))
            else:
                parts.append(self.project(x, a))
        return self.fc2(self.bn(torch.relu(torch.cat(parts))))
