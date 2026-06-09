import numpy as np
import torch

from src.basis import create_random_smooth_basis
from src.controllers import build_controller
from src.evaluation import compute_pca_trajectory, compute_trajectory_metrics
from src.render import render_skin


def test_basis_is_deterministic():
    a = create_random_smooth_basis(4, 16, 16, 4, seed=7)
    b = create_random_smooth_basis(4, 16, 16, 4, seed=7)
    assert torch.allclose(a, b)
    assert a.shape == (4, 1, 16, 16)


def test_render_skin_shape_and_range():
    basis = create_random_smooth_basis(4, 16, 16, 4, seed=7)
    weights = torch.zeros(4)
    skin = render_skin(weights, basis)
    assert skin.shape == (1, 16, 16)
    assert float(skin.min()) >= 0.0
    assert float(skin.max()) <= 1.0


def test_trajectory_metrics_are_finite():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(12, 5)).cumsum(axis=0)
    pca = compute_pca_trajectory(weights)
    metrics = compute_trajectory_metrics(weights, pca_coords=pca)
    assert pca.shape == (12, 2)
    assert metrics["pca_path_length"] > 0
    assert np.isfinite(metrics["pca_curvature_proxy_rad"])


def test_intermittent_decay_controller_records_sampling():
    weights = torch.zeros(3, requires_grad=True)
    controller = build_controller(
        weights,
        {
            "type": "intermittent_decay",
            "learning_rate": 0.1,
            "sampling": "fixed",
            "sample_interval": 2,
            "decay_tau": 2.0,
            "motor_momentum": 0.0,
        },
        seed=1,
    )

    for step in range(4):
        needs_gradient = controller.wants_gradient(step)
        if needs_gradient:
            weights.grad = torch.ones_like(weights)
        controller.step(step)

    records = controller.feedback_records
    assert [row["sample_event"] for row in records] == [1, 0, 1, 0]
    assert records[0]["feedback_gain"] == 1.0
    assert 0.0 < records[1]["feedback_gain"] < 1.0
