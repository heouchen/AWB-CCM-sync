from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from .metrics import relative_rgb_rmse, white_point_error
from .models import (
    AWBSyncModel,
    CCMSyncModel,
    awb_matrix_from_white_point,
    fit_local_ccm_matrix,
    sync_runtime,
)


@dataclass(frozen=True)
class SyntheticDataset:
    train_main_white_points: np.ndarray
    train_sub_white_points: np.ndarray
    train_main_rgb: np.ndarray
    train_sub_rgb: np.ndarray
    validation_main_white_points: np.ndarray
    validation_sub_white_points: np.ndarray
    validation_main_rgb: np.ndarray
    validation_sub_rgb: np.ndarray
    main_ccm: np.ndarray


@dataclass(frozen=True)
class SimulationReport:
    train_illuminants: int
    validation_illuminants: int
    patches_per_illuminant: int
    noise_std: float
    awb_naive_mean_error: float
    awb_sync_mean_error: float
    awb_sync_p95_error: float
    color_rmse_no_sync: float
    color_rmse_awb_only: float
    color_rmse_synced: float
    color_improvement_vs_awb_only: float
    max_condition_number: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _patches() -> np.ndarray:
    return np.array(
        [
            [0.38, 1.0, 0.30],
            [0.55, 1.0, 0.42],
            [0.74, 1.0, 0.55],
            [0.95, 1.0, 0.72],
            [1.18, 1.0, 0.90],
            [1.42, 1.0, 1.10],
            [0.45, 1.0, 0.78],
            [0.62, 1.0, 1.18],
            [0.82, 1.0, 1.48],
            [1.08, 1.0, 1.38],
            [1.35, 1.0, 1.20],
            [1.62, 1.0, 0.88],
            [0.52, 1.0, 1.55],
            [0.72, 1.0, 0.95],
            [0.92, 1.0, 0.48],
            [1.20, 1.0, 0.40],
            [1.55, 1.0, 0.58],
            [1.78, 1.0, 0.82],
            [0.30, 1.0, 0.35],
            [0.55, 1.0, 0.55],
            [0.78, 1.0, 0.78],
            [1.00, 1.0, 1.00],
            [1.25, 1.0, 1.25],
            [1.55, 1.0, 1.55],
        ],
        dtype=float,
    )


def _white_point_grid(u_values: np.ndarray, v_values: np.ndarray) -> np.ndarray:
    return np.array([[u, v] for u in u_values for v in v_values], dtype=float)


def _true_sub_white_point(main_white_point: np.ndarray) -> np.ndarray:
    u, v = main_white_point
    return np.array(
        [
            0.13 + 0.92 * u + 0.04 * v,
            0.10 + 0.05 * u + 0.88 * v,
        ],
        dtype=float,
    )


def _true_mapping(main_white_point: np.ndarray, sub_white_point: np.ndarray) -> np.ndarray:
    p1_u, p1_v = main_white_point
    p2_u, p2_v = sub_white_point
    m11 = 0.90 + 0.035 * p2_u - 0.012 * p2_v
    m31 = 0.030 + 0.010 * p2_u
    m13 = 0.025 - 0.008 * p2_v
    m33 = 0.94 + 0.015 * p2_u - 0.020 * p2_v
    m21 = p2_u - m11 * p1_u - m31 * p1_v
    m23 = p2_v - m13 * p1_u - m33 * p1_v
    return np.array(
        [
            [m11, 0.0, m13],
            [m21, 1.0, m23],
            [m31, 0.0, m33],
        ],
        dtype=float,
    )


def _main_ccm() -> np.ndarray:
    return np.array(
        [
            [1.14, -0.08, -0.06],
            [-0.04, 1.10, -0.06],
            [-0.03, -0.11, 1.14],
        ],
        dtype=float,
    )


def _camera_rgb(
    main_white_points: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _patches()
    sub_white_points = np.array([_true_sub_white_point(point) for point in main_white_points])
    main_rgb = []
    sub_rgb = []
    for main_wp, sub_wp in zip(main_white_points, sub_white_points):
        main = base * np.array([main_wp[0], 1.0, main_wp[1]])
        sub = main @ _true_mapping(main_wp, sub_wp)
        if noise_std > 0:
            main = np.maximum(main + rng.normal(0.0, noise_std, size=main.shape), 1e-6)
            sub = np.maximum(sub + rng.normal(0.0, noise_std, size=sub.shape), 1e-6)
        main_rgb.append(main)
        sub_rgb.append(sub)
    return sub_white_points, np.asarray(main_rgb), np.asarray(sub_rgb)


def generate_synthetic_dataset(*, noise_std: float = 0.0, seed: int = 7) -> SyntheticDataset:
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    rng = np.random.default_rng(seed)
    train_main_wp = _white_point_grid(np.linspace(0.70, 1.35, 6), np.linspace(0.60, 1.20, 6))
    validation_main_wp = _white_point_grid(np.linspace(0.765, 1.285, 5), np.linspace(0.66, 1.14, 5))
    train_sub_wp, train_main_rgb, train_sub_rgb = _camera_rgb(
        train_main_wp,
        noise_std=noise_std,
        rng=rng,
    )
    validation_sub_wp, validation_main_rgb, validation_sub_rgb = _camera_rgb(
        validation_main_wp,
        noise_std=noise_std,
        rng=rng,
    )
    return SyntheticDataset(
        train_main_white_points=train_main_wp,
        train_sub_white_points=train_sub_wp,
        train_main_rgb=train_main_rgb,
        train_sub_rgb=train_sub_rgb,
        validation_main_white_points=validation_main_wp,
        validation_sub_white_points=validation_sub_wp,
        validation_main_rgb=validation_main_rgb,
        validation_sub_rgb=validation_sub_rgb,
        main_ccm=_main_ccm(),
    )


def run_synthetic_simulation(
    *,
    noise_std: float = 0.001,
    seed: int = 7,
    l2: float = 1e-8,
) -> SimulationReport:
    dataset = generate_synthetic_dataset(noise_std=noise_std, seed=seed)
    awb_model = AWBSyncModel.fit(
        dataset.train_main_white_points,
        dataset.train_sub_white_points,
        degree=2,
        l2=l2,
    )
    local_matrices = np.array(
        [
            fit_local_ccm_matrix(main_rgb, sub_rgb, l2=l2)
            for main_rgb, sub_rgb in zip(dataset.train_main_rgb, dataset.train_sub_rgb)
        ]
    )
    ccm_model = CCMSyncModel.fit(
        dataset.train_sub_white_points,
        local_matrices,
        degree=2,
        l2=l2,
    )

    predicted_sub_wp = awb_model.predict(dataset.validation_main_white_points, clip=True)
    awb_errors = white_point_error(predicted_sub_wp, dataset.validation_sub_white_points)
    naive_awb_errors = white_point_error(
        dataset.validation_main_white_points,
        dataset.validation_sub_white_points,
    )

    main_outputs = []
    no_sync_outputs = []
    awb_only_outputs = []
    synced_outputs = []
    condition_numbers = []

    for idx, main_wp in enumerate(dataset.validation_main_white_points):
        main_awb = awb_matrix_from_white_point(main_wp)
        result = sync_runtime(
            main_wp,
            main_awb,
            dataset.main_ccm,
            awb_model,
            ccm_model,
            clip=True,
            cond_max=1e6,
            regularization=1e-8,
        )
        condition_numbers.append(float(np.linalg.cond(result.normalized_mapping @ result.sub_awb)))
        main_rgb = dataset.validation_main_rgb[idx]
        sub_rgb = dataset.validation_sub_rgb[idx]
        main_outputs.append(main_rgb @ main_awb @ dataset.main_ccm)
        no_sync_outputs.append(sub_rgb @ main_awb @ dataset.main_ccm)
        awb_only_outputs.append(sub_rgb @ result.sub_awb @ dataset.main_ccm)
        synced_outputs.append(sub_rgb @ result.sub_awb @ result.sub_ccm)

    main_stack = np.vstack(main_outputs)
    no_sync_stack = np.vstack(no_sync_outputs)
    awb_only_stack = np.vstack(awb_only_outputs)
    synced_stack = np.vstack(synced_outputs)
    color_rmse_awb_only = relative_rgb_rmse(awb_only_stack, main_stack)
    color_rmse_synced = relative_rgb_rmse(synced_stack, main_stack)

    return SimulationReport(
        train_illuminants=int(dataset.train_main_white_points.shape[0]),
        validation_illuminants=int(dataset.validation_main_white_points.shape[0]),
        patches_per_illuminant=int(dataset.validation_main_rgb.shape[1]),
        noise_std=float(noise_std),
        awb_naive_mean_error=float(np.mean(naive_awb_errors)),
        awb_sync_mean_error=float(np.mean(awb_errors)),
        awb_sync_p95_error=float(np.percentile(awb_errors, 95)),
        color_rmse_no_sync=relative_rgb_rmse(no_sync_stack, main_stack),
        color_rmse_awb_only=color_rmse_awb_only,
        color_rmse_synced=color_rmse_synced,
        color_improvement_vs_awb_only=float(color_rmse_awb_only / max(color_rmse_synced, 1e-12)),
        max_condition_number=float(np.max(condition_numbers)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic AWB/CCM sync simulation.")
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--l2", type=float, default=1e-8)
    args = parser.parse_args(argv)
    report = run_synthetic_simulation(noise_std=args.noise_std, seed=args.seed, l2=args.l2)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
