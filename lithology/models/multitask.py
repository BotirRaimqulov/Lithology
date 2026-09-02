"""Full pipeline: input projection -> 1D ResNet encoder -> sequence context
encoder -> {lithology, zone, boundary} heads.

    LAS curves (+ engineered features)
        |
    input projection (1x1 conv: n_features -> base_channels)
        |
    1D ResNet encoder (dilated residual CNN -- the receptive field / local
                        geological context; see resnet1d.py)
        |
    sequence context encoder (GRU or Transformer -- longer-range context
                               across the whole crop; optional)
        |
    +---------------+---------------+
    lithology head   zone head      boundary head
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from lithology.config import ModelConfig
from lithology.models.heads import BoundaryHead, LithologyHead, ZoneHead
from lithology.models.resnet1d import ResNet1DEncoder


class SinusoidalPositionalEncoding(nn.Module):
    """Encodes *relative position within the crop*, not absolute depth --
    the model must not be able to key off "this is depth X meters" (spec
    section 6/9), only off within-window order, which a Transformer needs
    explicitly since it has no other notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, L, C)."""
        return x + self.pe[:, : x.size(1)]


class SequenceContextEncoder(nn.Module):
    """Wraps an optional GRU/Transformer mixing stage on top of the ResNet
    features. Operates on and returns (B, C, L)."""

    def __init__(self, in_channels: int, kind: str, hidden_size: int, num_layers: int, num_heads: int,
                 dropout: float):
        super().__init__()
        self.kind = kind
        if kind == "none":
            self.out_channels = in_channels
        elif kind == "gru":
            self.gru = nn.GRU(
                in_channels, hidden_size, num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.out_channels = hidden_size * 2
        elif kind == "transformer":
            self.pos_enc = SinusoidalPositionalEncoding(in_channels)
            layer = nn.TransformerEncoderLayer(
                d_model=in_channels, nhead=num_heads, dim_feedforward=hidden_size,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.out_channels = in_channels
        else:
            raise ValueError(f"Unknown sequence_encoder {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, L) -> (B, out_channels, L)."""
        if self.kind == "none":
            return x
        x = x.transpose(1, 2)  # (B, L, C)
        if self.kind == "gru":
            x, _ = self.gru(x)
        else:
            x = self.transformer(self.pos_enc(x))
        return x.transpose(1, 2)


class MultiTaskLithologyModel(nn.Module):
    def __init__(self, in_features: int, config: ModelConfig):
        super().__init__()
        if config.num_classes_lithology is None or config.num_classes_zone is None:
            raise ValueError(
                "config.num_classes_lithology / num_classes_zone must be resolved from the "
                "training label vocabulary before constructing the model."
            )
        if config.encoder != "resnet1d":
            raise ValueError(f"Unknown model.encoder {config.encoder!r}")

        self.input_proj = nn.Conv1d(in_features, config.base_channels, kernel_size=1)
        self.encoder = ResNet1DEncoder(
            in_channels=config.base_channels,
            base_channels=config.base_channels,
            num_blocks=config.num_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.sequence_encoder = SequenceContextEncoder(
            in_channels=self.encoder.out_channels,
            kind=config.sequence_encoder,
            hidden_size=config.sequence_hidden_size,
            num_layers=config.sequence_num_layers,
            num_heads=config.sequence_num_heads,
            dropout=config.dropout,
        )
        c = self.sequence_encoder.out_channels
        self.lithology_head = LithologyHead(c, config.num_classes_lithology, dropout=config.dropout)
        self.zone_head = ZoneHead(c, config.num_classes_zone, dropout=config.dropout)
        self.boundary_head = BoundaryHead(c, dropout=config.dropout)

    def forward(self, features: torch.Tensor) -> dict:
        """``features``: (B, F, L). Returns per-point logits for all 3 heads."""
        x = self.input_proj(features)
        x = self.encoder(x)
        x = self.sequence_encoder(x)
        return {
            "lithology_logits": self.lithology_head(x),  # (B, L, num_lithology)
            "zone_logits": self.zone_head(x),             # (B, L, num_zone)
            "boundary_logits": self.boundary_head(x),     # (B, L)
        }

    def receptive_field_points(self) -> int:
        return self.encoder.receptive_field_points()
