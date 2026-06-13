"""
PINN Model — Physics-Informed Neural Network
for the damped harmonic oscillator.
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Fully-connected network that maps t -> x(t).

    Architecture: [1] -> [H]*n_layers -> [1]
    Activation  : Tanh (smooth, allows double-differentiation)
    """

    def __init__(self, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)
