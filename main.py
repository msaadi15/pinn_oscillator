#!/usr/bin/env python3
"""
main.py — run PINN training and generate all result figures + animation GIF.
usage
-----
    python main.py                        # default parameters
    python main.py --omega0 3 --gamma 0.5 --epochs 8000
"""

import argparse
import copy
import numpy as np
import torch

from src.model import PINN
from src.losses import total_loss
from src.utils import exact_solution, predict, relative_l2_error
from src.plot import plot_solution, plot_loss, make_training_gif


def parse_args():
    p = argparse.ArgumentParser(description="PINN — Damped Harmonic Oscillator")
    p.add_argument("--omega0",        type=float, default=2.0,  help="Natural frequency")
    p.add_argument("--gamma",         type=float, default=0.3,  help="Damping coefficient")
    p.add_argument("--t_max",         type=float, default=10.0, help="Time domain [0, t_max]")
    p.add_argument("--n_coll",        type=int,   default=200,  help="Collocation points")
    p.add_argument("--hidden",        type=int,   default=64,   help="Hidden units per layer")
    p.add_argument("--n_layers",      type=int,   default=4,    help="Number of hidden layers")
    p.add_argument("--epochs",        type=int,   default=5000, help="Training epochs")
    p.add_argument("--lr",            type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--lambda_ic",     type=float, default=10.0, help="IC loss weight")
    p.add_argument("--gif_snapshots", type=int,   default=60,   help="Frames in training GIF")
    p.add_argument("--save_dir",      type=str,   default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Config : ω₀={args.omega0}  γ={args.gamma}  epochs={args.epochs}\n")

    # ── Build model ────────────────────────────────────────────────────────
    model = PINN(hidden=args.hidden, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5, min_lr=1e-6
    )

    history  = {"physics": [], "ic": [], "total": []}
    snapshots = []                        # (epoch, state_dict, loss_info) for GIF

    snap_every = max(1, args.epochs // args.gif_snapshots)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t_coll = torch.rand(args.n_coll, 1, device=device) * args.t_max
        optimizer.zero_grad()
        loss, info = total_loss(model, t_coll, args.omega0, args.gamma, args.lambda_ic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        for k in history:
            history[k].append(info[k])

        if epoch % snap_every == 0 or epoch == 1:
            snapshots.append((epoch, copy.deepcopy(model.state_dict()), info))

        if epoch % 500 == 0:
            print(
                f"  Epoch {epoch:5d}  total={info['total']:.3e}  "
                f"physics={info['physics']:.3e}  ic={info['ic']:.3e}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

    print("\nTraining complete.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    t_np    = np.linspace(0, args.t_max, 500)
    x_exact = exact_solution(t_np, args.omega0, args.gamma)
    x_pred  = predict(model, t_np, device)
    l2_err  = relative_l2_error(x_pred, x_exact)
    print(f"Relative L² error : {l2_err:.4e}")

    # ── Save model ────────────────────────────────────────────────────────
    import pathlib
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = f"{args.save_dir}/pinn_omega{args.omega0}_gamma{args.gamma}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Model checkpoint  : {ckpt}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_solution(t_np, x_pred, x_exact, args.omega0, args.gamma, l2_err,
                  save_path=f"{args.save_dir}/solution.png")

    plot_loss(history, save_path=f"{args.save_dir}/loss.png")

    # ── GIF ───────────────────────────────────────────────────────────────
    print("\nGenerating training animation GIF…")
    make_training_gif(
        snapshots, args.omega0, args.gamma,
        t_max=args.t_max,
        save_path=f"{args.save_dir}/training.gif",
        fps=10,
    )
    print("\n✓ All outputs saved to:", args.save_dir)


if __name__ == "__main__":
    main()
