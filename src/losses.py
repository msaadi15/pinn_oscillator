"""
Physics-based loss functions for the damped harmonic oscillator.

Equation : x''(t) + 2*gamma*x'(t) + omega0^2 * x(t) = 0
IC        : x(0) = x0,  x'(0) = v0
"""

import torch


def ode_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    omega0: float,
    gamma: float,
) -> torch.Tensor:
    """
    Compute the ODE residual at collocation points.

    Uses torch.autograd.grad for exact automatic differentiation
    (the PyTorch equivalent of the finite-difference approach).
    """
    t = t.requires_grad_(True)
    x = model(t)

    # First derivative  dx/dt
    x_t = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
    )[0]

    # Second derivative  d²x/dt²
    x_tt = torch.autograd.grad(
        x_t, t,
        grad_outputs=torch.ones_like(x_t),
        create_graph=True,
    )[0]

    residual = x_tt + 2 * gamma * x_t + omega0 ** 2 * x
    return residual


def initial_condition_loss(
    model: torch.nn.Module,
    x0: float = 1.0,
    v0: float = 0.0,
) -> torch.Tensor:
    """
    Penalise violations of the initial conditions x(0) and x'(0).
    """
    t0 = torch.tensor([[0.0]], requires_grad=True)
    x = model(t0)

    x_t = torch.autograd.grad(
        x, t0,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
    )[0]

    loss_x = (x - x0) ** 2
    loss_v = (x_t - v0) ** 2
    return loss_x + loss_v


def total_loss(
    model: torch.nn.Module,
    t_coll: torch.Tensor,
    omega0: float,
    gamma: float,
    lambda_ic: float = 10.0,
    x0: float = 1.0,
    v0: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Weighted sum of physics residual + IC loss.

    Returns (total_loss, loss_dict) so each component can be logged.
    """
    res = ode_residual(model, t_coll, omega0, gamma)
    l_phy = (res ** 2).mean()

    l_ic = initial_condition_loss(model, x0, v0).squeeze()

    total = l_phy + lambda_ic * l_ic
    return total, {"physics": l_phy.item(), "ic": l_ic.item(), "total": total.item()}
