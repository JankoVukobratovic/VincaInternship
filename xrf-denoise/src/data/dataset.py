"""PyTorch Dataset for Poisson-split XRF denoising training."""

import numpy as np
import torch
from torch.utils.data import Dataset
from .poisson_split import poisson_split


def make_spatial_split(
    rows: int,
    cols: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    block_size: int = 15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Split pixel indices into train/val/test using spatial blocks.

    Divides the grid into blocks and assigns entire blocks to splits.
    This prevents spatial leakage between splits.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    block_size : int
        Size of spatial blocks (pixels).
    seed : int

    Returns
    -------
    dict with 'train', 'val', 'test' keys, each a 1D array of flat indices.
    """
    rng = np.random.default_rng(seed)

    # Create block labels
    n_blocks_r = (rows + block_size - 1) // block_size
    n_blocks_c = (cols + block_size - 1) // block_size
    n_blocks = n_blocks_r * n_blocks_c

    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)

    n_train = int(n_blocks * train_frac)
    n_val = int(n_blocks * val_frac)

    train_blocks = set(block_ids[:n_train])
    val_blocks = set(block_ids[n_train:n_train + n_val])
    test_blocks = set(block_ids[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []

    for r in range(rows):
        for c in range(cols):
            br = r // block_size
            bc = c // block_size
            bid = br * n_blocks_c + bc
            flat = r * cols + c

            if bid in train_blocks:
                train_idx.append(flat)
            elif bid in val_blocks:
                val_idx.append(flat)
            else:
                test_idx.append(flat)

    return {
        'train': np.array(train_idx),
        'val': np.array(val_idx),
        'test': np.array(test_idx),
    }


class XRFPoissonDataset(Dataset):
    """
    PyTorch Dataset that generates Poisson-split pairs on the fly.

    Each __getitem__ call creates a FRESH random split of the selected pixel,
    providing natural data augmentation — every epoch sees different noise.

    Parameters
    ----------
    spectra : np.ndarray, shape (N, C)
        Raw photon count spectra (integer or near-integer).
    indices : np.ndarray, shape (M,)
        Flat pixel indices to use (train/val/test split).
    global_scale : float
        Divide spectra by this value to keep inputs O(1).
    seed : int
        Base seed for reproducibility. Each worker gets seed + worker_id.
    """

    def __init__(
        self,
        spectra: np.ndarray,
        indices: np.ndarray,
        global_scale: float = 1.0,
        seed: int = 42,
    ):
        self.spectra = spectra  # (N, C)
        self.indices = indices
        self.global_scale = global_scale
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_idx = self.indices[idx]
        spectrum = self.spectra[pixel_idx]  # (C,)

        # Fresh Poisson split — different every call
        split_a, split_b = poisson_split(spectrum, self.rng)

        # Normalize by global_scale (per-channel scale, not total sum)
        input_tensor = torch.from_numpy(split_a / self.global_scale).unsqueeze(0)  # (1, C)
        target_tensor = torch.from_numpy(split_b / self.global_scale).unsqueeze(0)  # (1, C)

        return input_tensor, target_tensor
