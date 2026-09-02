import pytest

torch = pytest.importorskip("torch")

from lithology.config import ModelConfig
from lithology.constants import IGNORE_INDEX
from lithology.models.losses import MultiTaskLoss
from lithology.models.multitask import MultiTaskLithologyModel
from lithology.models.resnet1d import receptive_field_points


@pytest.mark.parametrize("sequence_encoder", ["none", "gru", "transformer"])
def test_forward_and_backward(sequence_encoder):
    cfg = ModelConfig(num_classes_lithology=4, num_classes_zone=3, sequence_encoder=sequence_encoder,
                       base_channels=16, num_blocks=(1, 1), sequence_hidden_size=16)
    model = MultiTaskLithologyModel(in_features=10, config=cfg)
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out["lithology_logits"].shape == (2, 64, 4)
    assert out["zone_logits"].shape == (2, 64, 3)
    assert out["boundary_logits"].shape == (2, 64)

    batch = {
        "lithology_label": torch.randint(0, 4, (2, 64)),
        "zone_label": torch.randint(0, 3, (2, 64)),
        "boundary_label": torch.randint(0, 2, (2, 64)),
    }
    loss_fn = MultiTaskLoss()
    total, parts = loss_fn(out, batch)
    total.backward()
    assert total.item() > 0


def test_output_length_matches_input_length_regardless_of_crop_size():
    cfg = ModelConfig(num_classes_lithology=2, num_classes_zone=2, base_channels=8, num_blocks=(1,))
    model = MultiTaskLithologyModel(in_features=5, config=cfg)
    for length in (17, 200, 513):
        out = model(torch.randn(1, 5, length))
        assert out["lithology_logits"].shape[1] == length


def test_ignore_index_excluded_from_loss():
    cfg = ModelConfig(num_classes_lithology=3, num_classes_zone=2, base_channels=8, num_blocks=(1,))
    model = MultiTaskLithologyModel(in_features=4, config=cfg)
    x = torch.randn(1, 4, 32)
    out = model(x)
    batch = {
        "lithology_label": torch.full((1, 32), IGNORE_INDEX, dtype=torch.long),
        "zone_label": torch.full((1, 32), IGNORE_INDEX, dtype=torch.long),
        "boundary_label": torch.full((1, 32), IGNORE_INDEX, dtype=torch.long),
    }
    loss_fn = MultiTaskLoss()
    total, parts = loss_fn(out, batch)
    # cross_entropy with all-ignored targets returns nan; boundary returns 0 (masked out).
    assert parts["loss_boundary"] == 0.0


def test_receptive_field_grows_with_more_blocks():
    small = receptive_field_points(kernel_size=7, num_blocks=(1,))
    large = receptive_field_points(kernel_size=7, num_blocks=(1, 1, 1, 1))
    assert large > small
