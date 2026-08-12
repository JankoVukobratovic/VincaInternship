"""
model.py - residual restoration U-Net for the neurips-restore MVP (item 3).

A small 2D U-Net (PyTorch, CPU-sized: ~0.5M parameters) that REFINES the
deterministic physics inversion, never replaces it.  It operates in the
FRONTAL frame (60 x 120) on all 8 element channels jointly - the
cross-element correlations of real pigments are legitimate prior
knowledge - plus two conditioning channels:

    channel 0..7   deterministic inverse of the tilted maps, one per
                   element line, normalized by fixed per-element scales
                   (P99 of prova1, see datagen.norm_scales)
    channel 8      validity: warped footprint / acquisition mask in
                   [0, 1] (0 = no data: outside the tilted footprint or
                   inside a simulated missing block)
    channel 9      constant angle channel, angle_deg / 25

The network predicts a RESIDUAL in normalized units; the restoration is

    restored_norm = input_norm[0:8] + net(input)

so at initialization (zero-initialized head) the model is exactly the
deterministic baseline, and the absolute level - whose bias headroom is
already exhausted by the physics inversion - is protected by
construction: the net has to actively learn any level change.

Architecture: 3 resolution levels (60x120 -> 30x60 -> 15x30), double
3x3 conv blocks with GroupNorm + SiLU, MaxPool down, transposed-conv
up, skip connections, 1x1 zero-init head.  Base width 32 gives ~0.47M
parameters.
"""

import torch
import torch.nn as nn

IN_CHANNELS = 10   # 8 elements + validity + angle
OUT_CHANNELS = 8   # residual per element


class DoubleConv(nn.Module):
    """(3x3 conv -> GroupNorm -> SiLU) x 2."""

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        g = groups if cout % groups == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class RestorationUNet(nn.Module):
    """Residual U-Net; forward() returns the residual, restore() adds it.

    Input  (B, 10, 60, 120)  normalized [elements | validity | angle]
    Output (B,  8, 60, 120)  residual in normalized element units
    """

    def __init__(self, in_ch: int = IN_CHANNELS, out_ch: int = OUT_CHANNELS,
                 base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        # zero-init head: the untrained model IS the deterministic baseline
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    def restore(self, x):
        """Full restoration in normalized units: input elements + residual."""
        return x[:, :OUT_CHANNELS] + self.forward(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
