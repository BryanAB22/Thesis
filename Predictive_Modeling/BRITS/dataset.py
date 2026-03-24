import os
import glob
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_psv_files(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.psv")))
    if not files:
        raise FileNotFoundError(f"No .psv files found in: {folder}")
    return files


def split_files(files: List[str], train_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)

    n_train = max(1, int(len(files) * train_frac))
    n_train = min(n_train, len(files) - 1) if len(files) > 1 else 1

    train_files = files[:n_train]
    val_files = files[n_train:] if len(files) > 1 else files[:]
    return train_files, val_files


def read_patient_window(
    file_path: str,
    feat_cols: List[str],
    T: int,
    keep_last: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      df_window: windowed dataframe (all columns preserved)
      x: numpy array [T, F] with NaNs kept
    """
    df = pd.read_csv(file_path, sep="|", na_values=["", "NaN", "nan", "NA"], keep_default_na=True)

    if df.shape[0] == 0:
        raise ValueError(f"Empty file: {file_path}")

    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)

    if n > T:
        df_window = df.tail(T).reset_index(drop=True) if keep_last else df.head(T).reset_index(drop=True)
    elif n < T:
        pad_n = T - n
        pad = pd.DataFrame(np.nan, index=range(pad_n), columns=df.columns)
        df_window = pd.concat([pad, df.reset_index(drop=True)], axis=0, ignore_index=True)
    else:
        df_window = df.reset_index(drop=True)

    x = df_window[feat_cols].to_numpy(dtype=np.float32)
    return df_window, x


def compute_train_stats(
    file_paths: List[str],
    feat_cols: List[str],
    T: int,
    keep_last: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nan-aware mean/std over the training set only.
    """
    chunks = []
    for fp in file_paths:
        _, x = read_patient_window(fp, feat_cols=feat_cols, T=T, keep_last=keep_last)
        chunks.append(x)

    X = np.stack(chunks, axis=0)  # [N, T, F]

    mean = np.nanmean(X, axis=(0, 1))
    std = np.nanstd(X, axis=(0, 1))

    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std > 1e-8), std, 1.0).astype(np.float32)

    return mean, std


def normalize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_norm = x.copy()
    mask = np.isfinite(x_norm)
    x_norm[mask] = (x_norm[mask] - mean[np.where(mask)[1]]) / std[np.where(mask)[1]]
    return x_norm.astype(np.float32)


def denormalize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x * std[None, :]) + mean[None, :]


class PatientWindowDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        feat_cols: List[str],
        T: int = 48,
        keep_last: bool = True,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.file_paths = file_paths
        self.feat_cols = feat_cols
        self.T = T
        self.keep_last = keep_last
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp = self.file_paths[idx]
        _, x = read_patient_window(
            fp,
            feat_cols=self.feat_cols,
            T=self.T,
            keep_last=self.keep_last
        )

        x_raw = x.copy()

        if self.normalize:
            if self.mean is None or self.std is None:
                raise ValueError("Normalization requested but mean/std were not provided.")
            x = normalize_array(x, self.mean, self.std)

        return {
            "x": torch.tensor(x, dtype=torch.float32),         # [T, F], still contains NaN
            "x_raw": torch.tensor(x_raw, dtype=torch.float32), # [T, F]
            "file_path": fp,
        }