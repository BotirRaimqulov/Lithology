"""Multi-task loss: weighted sum of lithology CE + zone CE + boundary BCE.

    L = weight_lithology * L_lithology
      + weight_zone       * L_zone
      + weight_boundary   * L_boundary

All three weights are configurable (``training.weight_*``). Points with no
ground truth (``IGNORE_INDEX``) are excluded from every term -- including
the boundary term, where ``BCEWithLogitsLoss`` has no built-in
``ignore_index`` so it is masked out manually here.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from lithology.constants import IGNORE_INDEX


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_lithology: float = 1.0,
        weight_zone: float = 1.0,
        weight_boundary: float = 1.0,
        lithology_class_weights: Optional[torch.Tensor] = None,
        zone_class_weights: Optional[torch.Tensor] = None,
        boundary_pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weight_lithology = weight_lithology
        self.weight_zone = weight_zone
        self.weight_boundary = weight_boundary
        self.register_buffer("lithology_class_weights", lithology_class_weights, persistent=False)
        self.register_buffer("zone_class_weights", zone_class_weights, persistent=False)
        self.register_buffer("boundary_pos_weight", boundary_pos_weight, persistent=False)

    def forward(self, outputs: dict, batch: dict) -> tuple:
        litho_logits = outputs["lithology_logits"]   # (B, L, K1)
        zone_logits = outputs["zone_logits"]          # (B, L, K2)
        boundary_logits = outputs["boundary_logits"]  # (B, L)

        litho_loss = F.cross_entropy(
            litho_logits.reshape(-1, litho_logits.shape[-1]),
            batch["lithology_label"].reshape(-1),
            weight=self.lithology_class_weights,
            ignore_index=IGNORE_INDEX,
        )
        zone_loss = F.cross_entropy(
            zone_logits.reshape(-1, zone_logits.shape[-1]),
            batch["zone_label"].reshape(-1),
            weight=self.zone_class_weights,
            ignore_index=IGNORE_INDEX,
        )

        boundary_target = batch["boundary_label"]
        mask = boundary_target != IGNORE_INDEX
        if mask.any():
            pos_weight = self.boundary_pos_weight
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits[mask], boundary_target[mask].float(), pos_weight=pos_weight,
            )
        else:
            boundary_loss = torch.zeros((), device=boundary_logits.device)

        total = (
            self.weight_lithology * litho_loss
            + self.weight_zone * zone_loss
            + self.weight_boundary * boundary_loss
        )
        return total, {
            "loss_total": total.item(),
            "loss_lithology": litho_loss.item(),
            "loss_zone": zone_loss.item(),
            "loss_boundary": boundary_loss.item(),
        }
