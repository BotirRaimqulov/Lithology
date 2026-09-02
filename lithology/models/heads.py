"""Per-depth-point task heads, sharing the encoder's sequence output."""
from __future__ import annotations

import torch
from torch import nn


class DenseHead(nn.Module):
    """A small per-timestep MLP head applied to every position of an
    (B, C, L) sequence, producing (B, L, out_dim)."""

    def __init__(self, in_channels: int, out_dim: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, out_dim, kernel_size=1),
            )
        else:
            self.net = nn.Conv1d(in_channels, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, L) -> (B, L, out_dim)."""
        out = self.net(x)
        return out.transpose(1, 2)


class LithologyHead(DenseHead):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__(in_channels, num_classes, hidden=in_channels // 2, dropout=dropout)


class ZoneHead(DenseHead):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__(in_channels, num_classes, hidden=in_channels // 2, dropout=dropout)


class BoundaryHead(DenseHead):
    """Outputs a single logit per depth point (boundary probability via sigmoid)."""

    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__(in_channels, out_dim=1, hidden=in_channels // 2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)  # (B, L)
