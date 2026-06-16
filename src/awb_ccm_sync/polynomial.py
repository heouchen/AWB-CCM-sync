from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_points(points: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError("A single 2D point must have shape (2,).")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Points must be finite.")
    return arr


def polynomial_terms(points: np.ndarray, degree: int) -> np.ndarray:
    """Build total-degree 2D polynomial terms in documented order.

    For degree 2 the order is [1, u, v, u^2, uv, v^2].
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    pts = _as_points(points)
    u = pts[:, 0]
    v = pts[:, 1]
    terms: list[np.ndarray] = []
    for total_degree in range(degree + 1):
        for u_power in range(total_degree, -1, -1):
            v_power = total_degree - u_power
            terms.append((u**u_power) * (v**v_power))
    return np.column_stack(terms)


@dataclass(frozen=True)
class Polynomial2D:
    degree: int
    coefficients: np.ndarray
    input_min: np.ndarray
    input_max: np.ndarray

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        values: np.ndarray,
        *,
        degree: int = 2,
        sample_weight: np.ndarray | None = None,
        l2: float = 0.0,
    ) -> "Polynomial2D":
        pts = _as_points(points)
        y = np.asarray(values, dtype=float).reshape(-1)
        if y.shape[0] != pts.shape[0]:
            raise ValueError("values length must match points length.")
        if not np.all(np.isfinite(y)):
            raise ValueError("values must be finite.")
        if l2 < 0:
            raise ValueError("l2 must be non-negative.")

        x = polynomial_terms(pts, degree)
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if weight.shape[0] != pts.shape[0]:
                raise ValueError("sample_weight length must match points length.")
            if np.any(weight < 0) or not np.all(np.isfinite(weight)):
                raise ValueError("sample_weight must be finite and non-negative.")
            scale = np.sqrt(weight)
            x = x * scale[:, None]
            y = y * scale

        if l2 > 0:
            reg = np.eye(x.shape[1])
            reg[0, 0] = 0.0
            x = np.vstack([x, np.sqrt(l2) * reg])
            y = np.concatenate([y, np.zeros(reg.shape[0])])

        coef = np.linalg.lstsq(x, y, rcond=None)[0]

        return cls(
            degree=degree,
            coefficients=coef,
            input_min=pts.min(axis=0),
            input_max=pts.max(axis=0),
        )

    def evaluate(self, points: np.ndarray, *, clip: bool = False) -> np.ndarray:
        pts = _as_points(points)
        if clip:
            pts = np.clip(pts, self.input_min, self.input_max)
        x = polynomial_terms(pts, self.degree)
        return x @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "coefficients": self.coefficients.tolist(),
            "input_min": self.input_min.tolist(),
            "input_max": self.input_max.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Polynomial2D":
        return cls(
            degree=int(payload["degree"]),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            input_min=np.asarray(payload["input_min"], dtype=float),
            input_max=np.asarray(payload["input_max"], dtype=float),
        )
