"""1D U-Net for XRF spectral denoising (Experiment A: from scratch)."""

import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net for spectral denoising.

    Architecture:
      - 4 encoder blocks with max-pooling (downsampling 2x each)
      - Bottleneck
      - 4 decoder blocks with upsampling + skip connections
      - Output: ReLU (counts must be non-negative)

    Input/Output: (batch, 1, C) where C = number of energy channels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        n_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        kernels = [7, 5, 3, 3]  # Larger kernels in early layers

        for i in range(n_blocks):
            ch_out = base_filters * (2 ** i)
            k = kernels[i] if i < len(kernels) else 3
            self.encoders.append(ConvBlock1d(ch_in, ch_out, k, dropout))
            self.pools.append(nn.MaxPool1d(2))
            ch_in = ch_out

        # Bottleneck
        self.bottleneck = ConvBlock1d(ch_in, ch_in, 3, dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(n_blocks - 1, -1, -1):
            ch_out = base_filters * (2 ** i)
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='linear',
                                              align_corners=False))
            # Input: upsampled + skip connection (concatenated)
            self.decoders.append(ConvBlock1d(ch_in + ch_out, ch_out, 3, dropout))
            ch_in = ch_out

        # Output — Softplus ensures non-negative predictions,
        # required for Poisson NLL loss (log of negative = NaN)
        self.output_conv = nn.Conv1d(base_filters, 1, kernel_size=1)
        self.output_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, C)

        Returns
        -------
        torch.Tensor, shape (batch, 1, C)
        """
        # Pad input to be divisible by 2^n_blocks
        orig_len = x.shape[2]
        divisor = 2 ** self.n_blocks
        pad_len = (divisor - orig_len % divisor) % divisor
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len), mode='reflect')

        # Encoder
        skips = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for up, dec, skip in zip(self.upsamples, self.decoders,
                                 reversed(skips)):
            h = up(h)
            # Handle size mismatch from pooling
            if h.shape[2] != skip.shape[2]:
                h = nn.functional.pad(h, (0, skip.shape[2] - h.shape[2]))
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = self.output_act(self.output_conv(h))

        # Remove padding
        if pad_len > 0:
            h = h[:, :, :orig_len]

        return h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
