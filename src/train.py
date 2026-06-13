"""
Training loop for the PINN oscillator.
"""

import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional

from .model import PINN
from .losses import total_loss


def train(
    omega0: float = 2.0,
    gamma: float = 0.3,
    t_max: float = 10.0,
    n_collocation: int = 200,
    hidden: int = 64,
    n_layers: int = 4,
    epochs: int = 5000,
    lr: float = 1e-3,
    lambda_ic: float = 10.0,
    device: Optional[str] = None,
    save_dir: str = "results",
    verbose: bool = True,
) -> tuple[PINN, dict]:
    """
    Train a PINN to solve:  x'' + 2γx' + ω₀²x = 0
    with x(0)=1, x'(0)=0.

    Returns
    -------
    model      : trained PINN
    history    : dict with lists of per-epoch losses
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PINN(hidden=hidden, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5, min_lr=1e-6
    )

    history = {"physics": [], "ic": [], "total": []}

    # Collocation points (resample each epoch for robustness)
    for epoch in range(1, epochs + 1):
        t_coll = torch.rand(n_collocation, 1, device=device) * t_max

        optimizer.zero_grad()
        loss, info = total_loss(model, t_coll, omega0, gamma, lambda_ic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        for k in history:
            history[k].append(info[k])

        if verbose and epoch % 500 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"total={info['total']:.4e}  "
                f"physics={info['physics']:.4e}  "
                f"ic={info['ic']:.4e}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

    # Save model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(save_dir) / f"pinn_omega{omega0}_gamma{gamma}.pt"
    torch.save(model.state_dict(), ckpt)
    if verbose:
        print(f"\nModel saved → {ckpt}")

    return model, history
