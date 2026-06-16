from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .polynomial import Polynomial2D

_EPS = 1e-12
_CCM_KEYS = ("m11", "m13", "m21", "m23", "m31", "m33")


def _as_white_points(points: np.ndarray | list[float] | list[list[float]]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(points, dtype=float)
    single = arr.ndim == 1
    if single:
        if arr.shape[0] != 2:
            raise ValueError("A single white point must have shape (2,).")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("White points must have shape (n, 2).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("White points must be finite.")
    if np.any(arr <= 0):
        raise ValueError("White point ratios must be positive.")
    return arr, single


def _as_matrix3(name: str, matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2: float = 0.0,
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("target length must match features rows.")
    if l2 < 0:
        raise ValueError("l2 must be non-negative.")
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.shape[0] != x.shape[0]:
            raise ValueError("sample_weight length must match features rows.")
        if np.any(weight < 0) or not np.all(np.isfinite(weight)):
            raise ValueError("sample_weight must be finite and non-negative.")
        scale = np.sqrt(weight)
        x = x * scale[:, None]
        y = y * scale

    if l2 > 0:
        reg = np.sqrt(l2) * np.eye(x.shape[1])
        x = np.vstack([x, reg])
        y = np.concatenate([y, np.zeros(reg.shape[0])])

    return np.linalg.lstsq(x, y, rcond=None)[0]


def normalize_rgb(rgb: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float)
    single = arr.ndim == 1
    if single:
        if arr.shape[0] != 3:
            raise ValueError("A single RGB row must have shape (3,).")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("RGB data must have shape (n, 3).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("RGB data must be finite.")
    if np.any(arr[:, 1] <= eps):
        raise ValueError("G channel must be positive for G-normalization.")

    normalized = arr / arr[:, 1:2]
    return normalized[0] if single else normalized


def awb_matrix_from_white_point(white_point: np.ndarray | list[float]) -> np.ndarray:
    points, single = _as_white_points(white_point)
    matrices = np.zeros((points.shape[0], 3, 3), dtype=float)
    matrices[:, 0, 0] = 1.0 / points[:, 0]
    matrices[:, 1, 1] = 1.0
    matrices[:, 2, 2] = 1.0 / points[:, 1]
    return matrices[0] if single else matrices


@dataclass(frozen=True)
class AWBSyncModel:
    rg: Polynomial2D
    bg: Polynomial2D

    @classmethod
    def fit(
        cls,
        main_white_points: np.ndarray,
        sub_white_points: np.ndarray,
        *,
        degree: int = 2,
        sample_weight: np.ndarray | None = None,
        l2: float = 0.0,
    ) -> "AWBSyncModel":
        main, _ = _as_white_points(main_white_points)
        sub, _ = _as_white_points(sub_white_points)
        if main.shape != sub.shape:
            raise ValueError("main_white_points and sub_white_points must have the same shape.")
        return cls(
            rg=Polynomial2D.fit(main, sub[:, 0], degree=degree, sample_weight=sample_weight, l2=l2),
            bg=Polynomial2D.fit(main, sub[:, 1], degree=degree, sample_weight=sample_weight, l2=l2),
        )

    def predict(self, main_white_point: np.ndarray | list[float], *, clip: bool = False) -> np.ndarray:
        points, single = _as_white_points(main_white_point)
        predicted = np.column_stack(
            [
                self.rg.evaluate(points, clip=clip),
                self.bg.evaluate(points, clip=clip),
            ]
        )
        if np.any(predicted <= 0):
            raise ValueError("Predicted sub-camera white point is non-positive.")
        return predicted[0] if single else predicted

    def awb_matrix(self, main_white_point: np.ndarray | list[float], *, clip: bool = False) -> np.ndarray:
        return awb_matrix_from_white_point(self.predict(main_white_point, clip=clip))

    def to_dict(self) -> dict[str, Any]:
        return {"rg": self.rg.to_dict(), "bg": self.bg.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AWBSyncModel":
        return cls(
            rg=Polynomial2D.from_dict(payload["rg"]),
            bg=Polynomial2D.from_dict(payload["bg"]),
        )


def fit_local_ccm_matrix(
    main_rgb: np.ndarray,
    sub_rgb: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2: float = 0.0,
) -> np.ndarray:
    main = normalize_rgb(main_rgb)
    sub = normalize_rgb(sub_rgb)
    if main.shape != sub.shape:
        raise ValueError("main_rgb and sub_rgb must have the same shape.")

    features = np.column_stack([main[:, 0], np.ones(main.shape[0]), main[:, 2]])
    coef_rg = _weighted_ridge(features, sub[:, 0], sample_weight=sample_weight, l2=l2)
    coef_bg = _weighted_ridge(features, sub[:, 2], sample_weight=sample_weight, l2=l2)

    matrix = np.array(
        [
            [coef_rg[0], 0.0, coef_bg[0]],
            [coef_rg[1], 1.0, coef_bg[1]],
            [coef_rg[2], 0.0, coef_bg[2]],
        ],
        dtype=float,
    )
    return matrix


@dataclass(frozen=True)
class CCMSyncModel:
    coefficients: dict[str, Polynomial2D]

    @classmethod
    def fit(
        cls,
        sub_white_points: np.ndarray,
        local_matrices: np.ndarray,
        *,
        degree: int = 2,
        sample_weight: np.ndarray | None = None,
        l2: float = 0.0,
    ) -> "CCMSyncModel":
        points, _ = _as_white_points(sub_white_points)
        matrices = np.asarray(local_matrices, dtype=float)
        if matrices.shape != (points.shape[0], 3, 3):
            raise ValueError("local_matrices must have shape (n, 3, 3).")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("local_matrices must be finite.")

        targets = {
            "m11": matrices[:, 0, 0],
            "m13": matrices[:, 0, 2],
            "m21": matrices[:, 1, 0],
            "m23": matrices[:, 1, 2],
            "m31": matrices[:, 2, 0],
            "m33": matrices[:, 2, 2],
        }
        return cls(
            coefficients={
                key: Polynomial2D.fit(points, value, degree=degree, sample_weight=sample_weight, l2=l2)
                for key, value in targets.items()
            }
        )

    def predict(self, sub_white_point: np.ndarray | list[float], *, clip: bool = False) -> np.ndarray:
        points, single = _as_white_points(sub_white_point)
        values = {key: self.coefficients[key].evaluate(points, clip=clip) for key in _CCM_KEYS}
        matrices = np.zeros((points.shape[0], 3, 3), dtype=float)
        matrices[:, 0, 0] = values["m11"]
        matrices[:, 0, 2] = values["m13"]
        matrices[:, 1, 0] = values["m21"]
        matrices[:, 1, 1] = 1.0
        matrices[:, 1, 2] = values["m23"]
        matrices[:, 2, 0] = values["m31"]
        matrices[:, 2, 2] = values["m33"]
        return matrices[0] if single else matrices

    def to_dict(self) -> dict[str, Any]:
        return {"coefficients": {key: model.to_dict() for key, model in self.coefficients.items()}}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CCMSyncModel":
        return cls(
            coefficients={
                key: Polynomial2D.from_dict(value)
                for key, value in payload["coefficients"].items()
            }
        )


def _ridge_solve(left: np.ndarray, right: np.ndarray, regularization: float) -> np.ndarray:
    return np.linalg.solve(
        left.T @ left + regularization * np.eye(left.shape[1]),
        left.T @ right,
    )


def solve_ccm2(
    main_awb: np.ndarray,
    main_ccm: np.ndarray,
    sub_awb: np.ndarray,
    normalized_mapping: np.ndarray,
    *,
    cond_max: float = 1e4,
    regularization: float = 0.0,
    fallback_ccm: np.ndarray | None = None,
) -> np.ndarray:
    awb1 = _as_matrix3("main_awb", main_awb)
    ccm1 = _as_matrix3("main_ccm", main_ccm)
    awb2 = _as_matrix3("sub_awb", sub_awb)
    mapping = _as_matrix3("normalized_mapping", normalized_mapping)
    if cond_max <= 0:
        raise ValueError("cond_max must be positive.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    left = mapping @ awb2
    right = awb1 @ ccm1
    try:
        cond = np.linalg.cond(left)
    except np.linalg.LinAlgError:
        cond = np.inf

    if not np.isfinite(cond) or cond > cond_max:
        if fallback_ccm is not None:
            return _as_matrix3("fallback_ccm", fallback_ccm).copy()
        if regularization <= 0:
            raise ValueError(f"M * AWB2 is ill-conditioned: condition number {cond:.3g}.")
        return _ridge_solve(left, right, regularization)

    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        if fallback_ccm is not None:
            return _as_matrix3("fallback_ccm", fallback_ccm).copy()
        if regularization <= 0:
            raise
        return _ridge_solve(left, right, regularization)


@dataclass(frozen=True)
class SyncResult:
    sub_white_point: np.ndarray
    sub_awb: np.ndarray
    normalized_mapping: np.ndarray
    sub_ccm: np.ndarray


def sync_runtime(
    main_white_point: np.ndarray | list[float],
    main_awb: np.ndarray,
    main_ccm: np.ndarray,
    awb_model: AWBSyncModel,
    ccm_model: CCMSyncModel,
    *,
    clip: bool = True,
    cond_max: float = 1e4,
    regularization: float = 0.0,
    fallback_ccm: np.ndarray | None = None,
) -> SyncResult:
    sub_white_point = awb_model.predict(main_white_point, clip=clip)
    sub_awb = awb_matrix_from_white_point(sub_white_point)
    normalized_mapping = ccm_model.predict(sub_white_point, clip=clip)
    sub_ccm = solve_ccm2(
        main_awb,
        main_ccm,
        sub_awb,
        normalized_mapping,
        cond_max=cond_max,
        regularization=regularization,
        fallback_ccm=fallback_ccm,
    )
    return SyncResult(
        sub_white_point=sub_white_point,
        sub_awb=sub_awb,
        normalized_mapping=normalized_mapping,
        sub_ccm=sub_ccm,
    )
