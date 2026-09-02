import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lithology.dataset.windowing import WellArrays
from lithology.training.train import evaluate


class _StubModel(torch.nn.Module):
    """Returns fixed logits regardless of input -- lets us deterministically
    simulate a healthy vs. diverged (NaN-producing) model."""

    def __init__(self, litho_logits, zone_logits, boundary_logits):
        super().__init__()
        self._litho = litho_logits
        self._zone = zone_logits
        self._boundary = boundary_logits

    def forward(self, x):
        b, _, length = x.shape
        return {
            "lithology_logits": self._litho.expand(b, length, -1),
            "zone_logits": self._zone.expand(b, length, -1),
            "boundary_logits": self._boundary.expand(b, length),
        }

    def eval(self):
        return self


def _make_well(n=20):
    return WellArrays(
        well_id="W",
        depth=np.arange(n, dtype=float) * 0.1,
        features=np.random.randn(n, 3).astype(np.float32),
        lithology_label=np.zeros(n, dtype=np.int64),
        zone_label=np.zeros(n, dtype=np.int64),
        boundary_label=np.zeros(n, dtype=np.int64),
        step=0.1,
    )


def test_evaluate_detects_diverged_nan_model():
    model = _StubModel(
        litho_logits=torch.full((1, 1, 3), float("nan")),
        zone_logits=torch.zeros(1, 1, 2),
        boundary_logits=torch.zeros(1, 1),
    )
    result = evaluate(model, [_make_well()], torch.device("cpu"), num_lithology=3, num_zone=2)
    assert result["diverged"] is True


def test_evaluate_healthy_model_not_flagged_diverged():
    model = _StubModel(
        litho_logits=torch.randn(1, 1, 3),
        zone_logits=torch.randn(1, 1, 2),
        boundary_logits=torch.randn(1, 1),
    )
    result = evaluate(model, [_make_well()], torch.device("cpu"), num_lithology=3, num_zone=2)
    assert result["diverged"] is False
