"""1D ResNet encoder for dense (per-depth-point) prediction.

This IS the "1D CNN" the spec talks about: a plain 1D CNN and a 1D ResNet
are not stacked as two separate stages here, because a residual CNN
already performs the convolutional feature extraction on its own -- adding
a second, separate CNN in front of it would be redundant (see the task
spec, section 10).

Unlike an image-classification ResNet, this network never downsamples the
sequence (no stride, no pooling): every depth point needs its own
lithology/zone/boundary prediction, so the output length must equal the
input length. The receptive field therefore grows through dilated
convolutions instead of striding -- each residual "stage" doubles the
dilation rate, which is the standard way (Temporal Convolutional Networks
/ WaveNet) to get a wide receptive field from small, cheap kernels rather
than one literal ``Conv1d(kernel_size=141)`` for a +/-7 m context.

:func:`receptive_field_points` lets the config's ``sampling.context_size``
be checked against what a given ``(kernel_size, num_blocks)`` choice
actually covers.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _same_padding(kernel_size: int, dilation: int) -> int:
    return ((kernel_size - 1) * dilation) // 2


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = _same_padding(kernel_size, dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.dropout(out)


class ResNet1DEncoder(nn.Module):
    """Stem conv + stacked dilated residual blocks, constant sequence length."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_blocks: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, padding=_same_padding(kernel_size, 1)),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        blocks = []
        for stage_idx, n_blocks_in_stage in enumerate(num_blocks):
            dilation = 2 ** stage_idx
            for _ in range(n_blocks_in_stage):
                blocks.append(ResidualBlock1D(base_channels, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = base_channels
        self.num_blocks = tuple(num_blocks)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, in_channels, length) -> (batch, out_channels, length)."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

    def receptive_field_points(self) -> int:
        return receptive_field_points(self.kernel_size, self.num_blocks)


def receptive_field_points(kernel_size: int, num_blocks: Sequence[int]) -> int:
    """Total receptive-field radius (in points, one-sided) of a
    :class:`ResNet1DEncoder` with the given kernel size and per-stage block
    counts (dilation doubles every stage, per-block RF contribution is
    ``2*(kernel_size-1)*dilation`` for two dilated convs per block, plus the
    stem's own ``kernel_size-1``).
    """
    radius = (kernel_size - 1) // 2  # stem
    for stage_idx, n_blocks_in_stage in enumerate(num_blocks):
        dilation = 2 ** stage_idx
        radius += n_blocks_in_stage * 2 * ((kernel_size - 1) // 2) * dilation
    return radius
