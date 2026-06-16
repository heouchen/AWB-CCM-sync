from __future__ import annotations

import numpy as np

from .models import normalize_rgb


def white_point_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(predicted, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if pred.shape != tgt.shape or pred.shape[-1] != 2:
        raise ValueError("predicted and target must have matching (..., 2) shapes.")
    return np.linalg.norm(pred - tgt, axis=-1)


def normalized_rgb_rmse(predicted_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    pred = normalize_rgb(predicted_rgb)
    tgt = normalize_rgb(target_rgb)
    if pred.shape != tgt.shape:
        raise ValueError("predicted_rgb and target_rgb must have the same shape.")
    return float(np.sqrt(np.mean((pred - tgt) ** 2)))


def relative_rgb_rmse(predicted_rgb: np.ndarray, target_rgb: np.ndarray, *, eps: float = 1e-12) -> float:
    pred = np.asarray(predicted_rgb, dtype=float)
    tgt = np.asarray(target_rgb, dtype=float)
    if pred.shape != tgt.shape or pred.shape[-1] != 3:
        raise ValueError("predicted_rgb and target_rgb must have matching (..., 3) shapes.")
    denom = float(np.sqrt(np.mean(tgt**2)))
    if denom <= eps:
        raise ValueError("target_rgb energy is too small for relative RMSE.")
    return float(np.sqrt(np.mean((pred - tgt) ** 2)) / denom)
