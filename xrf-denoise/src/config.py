"""Global configuration for the XRF denoising pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # ─── Data ────────────────────────────────────────────────────────────────
    n_channels: int = 1024          # Energy channels per spectrum (our MCA files)
    rows: int = 60                  # Scan grid rows
    cols: int = 120                 # Scan grid columns
    n_pixels: int = 7200            # rows * cols
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Detectors (our "pistol A" and "pistol B")
    detector_a: str = "10264"
    detector_b: str = "19511"

    # Datasets
    dataset_train: str = "aurora-antico1-prova1"  # Train/val/test on prova1
    dataset_val_external: str = "aurora-antico1-prova2"  # External validation

    # Elements to analyze (Sn EXCLUDED — artifact)
    elements: dict = field(default_factory=lambda: {
        'Ca':    {'kev': 3.69, 'name': 'Calcium Ka'},
        'Ti':    {'kev': 4.51, 'name': 'Titanium Ka'},
        'Fe':    {'kev': 6.40, 'name': 'Iron Ka'},
        'Cu':    {'kev': 8.05, 'name': 'Copper Ka'},
        'Pb_La': {'kev': 10.55, 'name': 'Lead La'},
    })

    # Energy calibration (from existing codebase)
    cal_slope: float = 0.0292       # keV per channel
    cal_intercept: float = -0.0044  # keV offset (approx from linregress)

    # ─── Model (Experiment A: from scratch) ──────────────────────────────────
    base_filters: int = 32
    n_encoder_blocks: int = 4
    dropout: float = 0.15

    # ─── Model (Experiment B: pretrained) ────────────────────────────────────
    pretrained_backbone: str = "resnet18"
    freeze_epochs: int = 20
    backbone_lr_factor: float = 0.1

    # ─── Training (shared) ───────────────────────────────────────────────────
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 50
    patience: int = 12              # Early stopping
    loss: str = "poisson_nll"        # "poisson_nll", "mse", or "mixed"
    mixed_alpha: float = 0.5        # Weight for MSE in mixed loss
    seed: int = 42

    # ─── Hardware ────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Paths (relative to xrf-denoise/) ────────────────────────────────────
    project_root: str = ""          # Set at runtime
    raw_data_dir: str = ""          # Set at runtime (parent dir with MCA files)
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    exp_a_dir: str = "experiments/A_scratch"
    exp_b_dir: str = "experiments/B_pretrained"
    comparison_dir: str = "experiments/comparison"
    figures_dir: str = "figures"

    def __post_init__(self):
        if not self.project_root:
            self.project_root = str(Path(__file__).parent.parent)
        if not self.raw_data_dir:
            # MCA files are in the parent VincaInternship directory
            self.raw_data_dir = str(Path(self.project_root).parent)

    def abs_path(self, relative: str) -> Path:
        """Convert a relative path to absolute within project root."""
        return Path(self.project_root) / relative

    def energy_axis(self) -> "np.ndarray":
        """Return energy axis in keV for all channels."""
        import numpy as np
        return np.arange(self.n_channels) * self.cal_slope + self.cal_intercept

    def kev_to_channel(self, kev: float) -> int:
        """Convert keV to channel index."""
        return int(round((kev - self.cal_intercept) / self.cal_slope))
