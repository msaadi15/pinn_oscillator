"""
Analytical solution and evaluation utilities.
"""

import numpy as np
import torch
from typing import Optional


def exact_solution(
    t: np.ndarray,
    omega0: float,
    gamma: float,
    x0: float = 1.0,
    v0: float = 0.0,
) -> np.ndarray:
    """
    Closed-form solution of x'' + 2γx' + ω₀²x = 0.

    Three regimes
    -------------
    under-damped  : γ < ω₀  →  decaying oscillation
    critically damped: γ = ω₀
    over-damped   : γ > ω₀  →  no oscillation
    """
    if gamma < omega0:                          # under-damped
        wd = np.sqrt(omega0 ** 2 - gamma ** 2)
        A = x0
        B = (v0 + gamma * x0) / wd
        return np.exp(-gamma * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))

    elif np.isclose(gamma, omega0):             # critically damped
        A = x0
        B = v0 + gamma * x0
        return (A + B * t) * np.exp(-gamma * t)

    else:                                       # over-damped
        s = np.sqrt(gamma ** 2 - omega0 ** 2)
        s1, s2 = -gamma + s, -gamma - s
        A = (v0 - s2 * x0) / (s1 - s2)
        B = (s1 * x0 - v0) / (s1 - s2)
        return A * np.exp(s1 * t) + B * np.exp(s2 * t)


def predict(
    model: torch.nn.Module,
    t_np: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run inference on a numpy time array and return numpy predictions."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_np[:, None], dtype=torch.float32, device=device)
        x_pred = model(t_tensor).cpu().numpy().flatten()
    return x_pred


def relative_l2_error(pred: np.ndarray, exact: np.ndarray) -> float:
    """Relative L2 error between PINN and exact solution."""
    return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12))
