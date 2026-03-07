"""
Experiment B: Pretrained backbone for XRF spectral denoising.

Option B1: Reshape 1D spectrum (1, 1024) into pseudo-2D (1, 32, 32)
and use a pretrained ResNet-18 encoder with a fresh decoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetSpectralDenoiser(nn.Module):
    """
    Pretrained ResNet-18 encoder + fresh decoder for spectral denoising.

    The trick: reshape 1D spectrum (1024,) into pseudo-2D image (1, 32, 32),
    pass through ResNet encoder (borrowing learned feature detectors),
    then decode back to 1D spectrum.

    The first conv layer is replaced to accept 1-channel input.
    """

    def __init__(self, n_channels: int = 1024, freeze_encoder: bool = True):
        super().__init__()
        self.n_channels = n_channels

        # Pseudo-2D reshape dimensions: 1024 = 32 x 32
        self.h2d = 32
        self.w2d = 32
        assert self.h2d * self.w2d == n_channels

        # ── Encoder: pretrained ResNet-18 ────────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace first conv: 3ch -> 1ch (grayscale spectrum "image")
        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
        # Initialize from pretrained: average across RGB channels
        with torch.no_grad():
            self.encoder_conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_pool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1  # 64ch, 8x8
        self.encoder_layer2 = resnet.layer2  # 128ch, 4x4
        self.encoder_layer3 = resnet.layer3  # 256ch, 2x2
        self.encoder_layer4 = resnet.layer4  # 512ch, 1x1

        if freeze_encoder:
            self._freeze_encoder()

        # ── Decoder: fresh transposed convolutions ───────────────────────
        self.decoder = nn.Sequential(
            # 512 x 1 x 1 -> 256 x 2 x 2
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 2 x 2 -> 128 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 4 x 4 -> 64 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 8 x 8 -> 32 x 16 x 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 x 16 x 16 -> 1 x 32 x 32
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # Non-negative counts
        )

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in [self.encoder_conv1, self.encoder_bn1,
                      self.encoder_layer1, self.encoder_layer2,
                      self.encoder_layer3, self.encoder_layer4]:
            for p in param.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self, lr_factor: float = 0.1):
        """Unfreeze encoder for fine-tuning."""
        for param in [self.encoder_conv1, self.encoder_bn1,
                      self.encoder_layer1, self.encoder_layer2,
                      self.encoder_layer3, self.encoder_layer4]:
            for p in param.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, 1024) — 1D spectrum

        Returns
        -------
        (batch, 1, 1024) — denoised 1D spectrum
        """
        batch_size = x.shape[0]

        # Reshape 1D -> pseudo-2D: (B, 1, 1024) -> (B, 1, 32, 32)
        h = x.view(batch_size, 1, self.h2d, self.w2d)

        # Encoder
        h = self.encoder_conv1(h)     # (B, 64, 16, 16)
        h = self.encoder_bn1(h)
        h = self.encoder_relu(h)
        h = self.encoder_pool(h)      # (B, 64, 8, 8)
        h = self.encoder_layer1(h)    # (B, 64, 8, 8)
        h = self.encoder_layer2(h)    # (B, 128, 4, 4)
        h = self.encoder_layer3(h)    # (B, 256, 2, 2)
        h = self.encoder_layer4(h)    # (B, 512, 1, 1)

        # Decoder
        h = self.decoder(h)           # (B, 1, 32, 32)

        # Reshape back to 1D: (B, 1, 32, 32) -> (B, 1, 1024)
        h = h.view(batch_size, 1, self.n_channels)

        return h

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
