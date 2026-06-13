"""
Tests for PINN components.
Run with: pytest tests/
"""
import pytest
import torch
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.model import PINN
from src.losses import ode_residual, initial_condition_loss, total_loss
from src.utils import exact_solution, predict, relative_l2_error


# ── Model ─────────────────────────────────────────────────────────────────────

class TestPINN:
    def test_output_shape(self):
        model = PINN(hidden=16, n_layers=2)
        t = torch.linspace(0, 1, 50).unsqueeze(-1)
        x = model(t)
        assert x.shape == (50, 1)

    def test_differentiable(self):
        model = PINN(hidden=16, n_layers=2)
        t = torch.tensor([[0.5]], requires_grad=True)
        x = model(t)
        grad = torch.autograd.grad(x, t)[0]
        assert grad.shape == t.shape

    def test_init_weights_no_nan(self):
        model = PINN(hidden=32, n_layers=3)
        for p in model.parameters():
            assert not torch.isnan(p).any()


# ── Losses ────────────────────────────────────────────────────────────────────

class TestLosses:
    def setup_method(self):
        self.model = PINN(hidden=16, n_layers=2)

    def test_ode_residual_shape(self):
        t = torch.linspace(0.01, 5.0, 20).unsqueeze(-1)
        res = ode_residual(self.model, t, omega0=2.0, gamma=0.3)
        assert res.shape == (20, 1)

    def test_ic_loss_non_negative(self):
        loss = initial_condition_loss(self.model, x0=1.0, v0=0.0)
        assert loss.item() >= 0

    def test_total_loss_returns_dict(self):
        t = torch.rand(10, 1)
        _, info = total_loss(self.model, t, omega0=2.0, gamma=0.3)
        assert set(info.keys()) == {"physics", "ic", "total"}
        assert info["total"] >= 0


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_exact_underdamped(self):
        t = np.array([0.0])
        x = exact_solution(t, omega0=2.0, gamma=0.3)
        assert np.isclose(x[0], 1.0, atol=1e-6)  # x(0) == 1

    def test_exact_critically_damped(self):
        t = np.array([0.0])
        x = exact_solution(t, omega0=1.0, gamma=1.0)
        assert np.isclose(x[0], 1.0, atol=1e-6)

    def test_exact_overdamped(self):
        t = np.array([0.0])
        x = exact_solution(t, omega0=1.0, gamma=2.0)
        assert np.isclose(x[0], 1.0, atol=1e-6)

    def test_predict_shape(self):
        model = PINN(hidden=16, n_layers=2)
        t = np.linspace(0, 5, 100)
        x = predict(model, t)
        assert x.shape == (100,)

    def test_l2_error_identical(self):
        x = np.ones(50)
        assert relative_l2_error(x, x) == pytest.approx(0.0, abs=1e-9)
