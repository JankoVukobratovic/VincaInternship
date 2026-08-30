"""
toy_model.py

A tiny residual CNN standing in for the real pipeline's small residual
U-Nets. 3 conv layers, in_channels=1 (simplified from the 2-channel
[degraded input + validity mask] design since this toy problem has no
missing-data mask), out_channels=1, a few thousand parameters, with the
final conv layer zero-initialised so an untrained network is exactly the
identity map (pure residual correction learned on top of that).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_forward import SimKnobs, nominal_knobs, jitter_knobs, random_field, forward

HIDDEN = 16


class ResCNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 1, 3, padding=1)
        # zero-init final layer -> net(x) == x at initialisation (identity)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        delta = self.conv3(h)
        return x + delta


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def train_one_net(kind: str, seed: int, n_steps: int = 300, batch_size: int = 10,
                   lr: float = 2e-3, verbose_every: int = 0):
    """Train one ResCNN.

    kind == 'jitter': draw ONE knob perturbation (within calibration
        uncertainty of nominal) for this network's seed, and train against
        that fixed, perturbed simulator throughout (fresh random fields
        every batch). This mirrors "an ensemble member trained on a
        randomly-perturbed simulator instance".
    kind == 'control': train against the fixed nominal simulator throughout.

    Returns (trained_net, knobs_used).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + (10_000 if kind == "jitter" else 20_000))

    if kind == "jitter":
        knobs = jitter_knobs(rng)
    elif kind == "control":
        knobs = nominal_knobs()
    else:
        raise ValueError(f"unknown kind: {kind}")

    net = ResCNN()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for step in range(n_steps):
        fields = [random_field(rng) for _ in range(batch_size)]
        obs = [forward(f, knobs, rng) for f in fields]
        x = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(np.stack(fields), dtype=torch.float32).unsqueeze(1)
        mask = torch.ones_like(y)  # no missing-data mask in this toy; kept for parity with "masked L1"

        pred = net(x)
        loss = (torch.abs(pred - y) * mask).sum() / mask.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose_every and (step + 1) % verbose_every == 0:
            print(f"    [{kind} seed={seed}] step {step + 1}/{n_steps} loss={loss.item():.4f}")

    return net, knobs


@torch.no_grad()
def predict(net: nn.Module, obs: np.ndarray) -> np.ndarray:
    net.eval()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = net(x)
    return y.squeeze(0).squeeze(0).numpy()
