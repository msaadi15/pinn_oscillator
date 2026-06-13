"""
Plotting utilities — solution curves, loss history, and training GIF.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Optional

from .utils import exact_solution

# ── Style ────────────────────────────────────────────────────────────────────
BLUE   = "#185FA5"
ORANGE = "#D85A30"
GREEN  = "#0F6E56"
GRAY   = "#888780"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


# ── Final solution plot ───────────────────────────────────────────────────────
def plot_solution(
    t: np.ndarray,
    x_pred: np.ndarray,
    x_exact: np.ndarray,
    omega0: float,
    gamma: float,
    l2_err: float,
    save_path: str = "results/solution.png",
):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, x_exact, color=BLUE,   lw=2,   ls="--", label="Exact solution")
    ax.plot(t, x_pred,  color=ORANGE, lw=2.5, label=f"PINN  (L² err={l2_err:.2e})")
    ax.fill_between(t, x_pred, x_exact, alpha=0.12, color=ORANGE)
    ax.set_xlabel("t", fontsize=13)
    ax.set_ylabel("x(t)", fontsize=13)
    ax.set_title(
        rf"Damped Harmonic Oscillator — $\omega_0={omega0}$, $\gamma={gamma}$",
        fontsize=14,
    )
    ax.legend(fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Loss curves ───────────────────────────────────────────────────────────────
def plot_loss(
    history: dict,
    save_path: str = "results/loss.png",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(history["total"]) + 1)
    ax.semilogy(epochs, history["total"],   color=ORANGE, lw=2,   label="Total loss")
    ax.semilogy(epochs, history["physics"], color=BLUE,   lw=1.5, label="Physics residual", ls="--")
    ax.semilogy(epochs, history["ic"],      color=GREEN,  lw=1.5, label="IC loss",           ls=":")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (log scale)", fontsize=13)
    ax.set_title("PINN Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Training animation (GIF) ─────────────────────────────────────────────────
def make_training_gif(
    model_snapshots: list,          # list of (epoch, state_dict)
    omega0: float,
    gamma: float,
    t_max: float = 10.0,
    n_pts: int = 300,
    save_path: str = "results/training.gif",
    fps: int = 12,
):
    """
    Render an animated GIF showing how the PINN prediction evolves
    epoch-by-epoch as training progresses.

    Parameters
    ----------
    model_snapshots : list of (epoch_number, state_dict) tuples
        Snapshots recorded every N epochs during training.
    """
    import torch
    from .model import PINN

    t_np     = np.linspace(0, t_max, n_pts)
    x_exact  = exact_solution(t_np, omega0, gamma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Trajectory panel ────────────────────────────────────────────────────
    ax1.plot(t_np, x_exact, color=BLUE, lw=2, ls="--", label="Exact", zorder=3)
    (line_pred,) = ax1.plot([], [], color=ORANGE, lw=2.5, label="PINN", zorder=4)
    fill_ref = ax1.fill_between([], [], [], alpha=0)

    ax1.set_xlim(0, t_max)
    y_pad = 0.25
    ax1.set_ylim(x_exact.min() - y_pad, x_exact.max() + y_pad)
    ax1.set_xlabel("t", fontsize=12)
    ax1.set_ylabel("x(t)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper right")
    title1 = ax1.set_title("", fontsize=13)

    # ── Loss panel ──────────────────────────────────────────────────────────
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Total loss (log)", fontsize=12)
    ax2.set_title("Training loss", fontsize=13)
    all_losses = [info["total"] for _, _, info in model_snapshots]
    ax2.set_xlim(0, model_snapshots[-1][0])
    ax2.set_ylim(
        max(1e-8, min(all_losses) * 0.5),
        max(all_losses) * 2,
    )
    ax2.set_yscale("log")
    (line_loss,) = ax2.plot([], [], color=GREEN, lw=2)

    fig.tight_layout()

    epochs_plot, losses_plot = [], []

    # Infer hidden size from first snapshot
    first_sd = model_snapshots[0][1]
    hidden_inferred = first_sd["net.0.bias"].shape[0]
    n_layers_inferred = sum(1 for k in first_sd if k.endswith(".weight") and "net" in k) - 1
    dummy = PINN(hidden=hidden_inferred, n_layers=n_layers_inferred)

    def _fill_between(ax, x, y1, y2, **kw):
        """Helper: clear previous fill and redraw."""
        for coll in ax.collections:
            coll.remove()
        ax.fill_between(x, y1, y2, **kw)

    def update(frame):
        epoch, sd, info = model_snapshots[frame]
        dummy.load_state_dict(sd)
        dummy.eval()
        with torch.no_grad():
            t_t   = torch.tensor(t_np[:, None], dtype=torch.float32)
            x_pred = dummy(t_t).numpy().flatten()

        line_pred.set_data(t_np, x_pred)
        _fill_between(ax1, t_np, x_pred, x_exact,
                      alpha=0.10, color=ORANGE, zorder=2)

        l2 = np.linalg.norm(x_pred - x_exact) / (np.linalg.norm(x_exact) + 1e-12)
        title1.set_text(
            rf"$\omega_0={omega0}$, $\gamma={gamma}$ — epoch {epoch}  (L²={l2:.2e})"
        )

        epochs_plot.append(epoch)
        losses_plot.append(info["total"])
        line_loss.set_data(epochs_plot, losses_plot)

        return line_pred, line_loss

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(model_snapshots),
        interval=1000 // fps,
        blit=False,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved GIF → {save_path}")
