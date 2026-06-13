from .model import PINN
from .losses import total_loss, ode_residual, initial_condition_loss
from .train import train
from .utils import exact_solution, predict, relative_l2_error
from .plot import plot_solution, plot_loss, make_training_gif

__all__ = [
    "PINN",
    "total_loss",
    "ode_residual",
    "initial_condition_loss",
    "train",
    "exact_solution",
    "predict",
    "relative_l2_error",
    "plot_solution",
    "plot_loss",
    "make_training_gif",
]
