from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

try:
    import colour
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
    raise ImportError(
        "Spectral simulation requires the 'colour-science' package. "
        "Install project dependencies from pyproject.toml before running it."
    ) from exc

from colour import SpectralShape

from .metrics import normalized_rgb_rmse, relative_rgb_rmse, white_point_error
from .models import (
    AWBSyncModel,
    CCMSyncModel,
    awb_matrix_from_white_point,
    fit_local_ccm_matrix,
    normalize_rgb,
    sync_runtime,
)

COLORCHECKER_NAME = "BabelColor Average"
MAIN_CAMERA_NAME = "Nikon 5100 (NPL)"
SUB_CAMERA_NAME = "Sigma SDMerill (NPL)"
GREY_PATCH_NAME = "neutral 8 (.23 D)"
SPECTRAL_SHAPE = SpectralShape(400, 700, 10)
VALIDATION_ILLUMINANTS = (
    "D55",
    "D75",
    "FL4",
    "FL10",
    "LED-B2",
    "LED-V1",
    "HP3",
    "ISO 7589 Photoflood",
)
TRAIN_ILLUMINANTS = tuple(
    illuminant_name
    for illuminant_name in colour.SDS_ILLUMINANTS.keys()
    if illuminant_name not in VALIDATION_ILLUMINANTS
)


@dataclass(frozen=True)
class SpectralDataset:
    train_illuminants: tuple[str, ...]
    validation_illuminants: tuple[str, ...]
    patch_names: tuple[str, ...]
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
class SpectralSimulationReport:
    colorchecker: str
    main_camera: str
    sub_camera: str
    wavelength_start_nm: int
    wavelength_end_nm: int
    wavelength_interval_nm: int
    train_illuminants: int
    validation_illuminants: int
    patches_per_illuminant: int
    noise_std: float
    awb_naive_mean_error: float
    awb_sync_mean_error: float
    awb_sync_p95_error: float
    chroma_rmse_no_sync: float
    chroma_rmse_awb_only: float
    chroma_rmse_synced: float
    color_rmse_no_sync: float
    color_rmse_awb_only: float
    color_rmse_synced: float
    color_improvement_vs_awb_only: float
    max_condition_number: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _aligned_sd_values(sd_name: str, mapping, *, shape: SpectralShape) -> np.ndarray:
    values = mapping[sd_name].copy().align(shape).values
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _camera_values(camera_name: str, *, shape: SpectralShape) -> np.ndarray:
    camera = colour.MSDS_CAMERA_SENSITIVITIES[camera_name].copy().align(shape)
    return np.maximum(np.asarray(camera.values, dtype=float), 0.0)


def _colourchecker_values(
    colorchecker_name: str,
    *,
    shape: SpectralShape,
) -> tuple[tuple[str, ...], np.ndarray]:
    colorchecker = colour.SDS_COLOURCHECKERS[colorchecker_name]
    patch_names = tuple(colorchecker.keys())
    reflectances = np.vstack(
        [
            np.maximum(np.asarray(colorchecker[name].copy().align(shape).values, dtype=float), 0.0)
            for name in patch_names
        ]
    )
    return patch_names, reflectances


def _virtual_capture(
    illuminant_names: tuple[str, ...],
    *,
    camera_name: str,
    patch_names: tuple[str, ...],
    reflectances: np.ndarray,
    grey_patch_name: str,
    shape: SpectralShape,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    camera = _camera_values(camera_name, shape=shape)
    grey_index = patch_names.index(grey_patch_name)
    captures = []
    for illuminant_name in illuminant_names:
        illuminant = _aligned_sd_values(illuminant_name, colour.SDS_ILLUMINANTS, shape=shape)
        spectral_product = illuminant[:, None] * camera
        rgb = reflectances @ spectral_product * shape.interval
        rgb = rgb / rgb[grey_index, 1]
        if noise_std > 0:
            rgb = np.maximum(rgb + rng.normal(0.0, noise_std, size=rgb.shape), 1e-9)
        captures.append(rgb)
    return np.asarray(captures)


def _white_points(rgb: np.ndarray, *, patch_names: tuple[str, ...], grey_patch_name: str) -> np.ndarray:
    grey_index = patch_names.index(grey_patch_name)
    grey = rgb[:, grey_index, :]
    return np.column_stack([grey[:, 0] / grey[:, 1], grey[:, 2] / grey[:, 1]])


def _main_ccm() -> np.ndarray:
    return np.array(
        [
            [1.12, -0.07, -0.05],
            [-0.05, 1.11, -0.06],
            [-0.03, -0.10, 1.13],
        ],
        dtype=float,
    )


def generate_spectral_dataset(
    *,
    noise_std: float = 0.0,
    seed: int = 13,
    shape: SpectralShape = SPECTRAL_SHAPE,
    colorchecker_name: str = COLORCHECKER_NAME,
    main_camera_name: str = MAIN_CAMERA_NAME,
    sub_camera_name: str = SUB_CAMERA_NAME,
    grey_patch_name: str = GREY_PATCH_NAME,
    train_illuminants: tuple[str, ...] = TRAIN_ILLUMINANTS,
    validation_illuminants: tuple[str, ...] = VALIDATION_ILLUMINANTS,
) -> SpectralDataset:
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    rng = np.random.default_rng(seed)
    patch_names, reflectances = _colourchecker_values(colorchecker_name, shape=shape)
    train_main_rgb = _virtual_capture(
        train_illuminants,
        camera_name=main_camera_name,
        patch_names=patch_names,
        reflectances=reflectances,
        grey_patch_name=grey_patch_name,
        shape=shape,
        noise_std=noise_std,
        rng=rng,
    )
    train_sub_rgb = _virtual_capture(
        train_illuminants,
        camera_name=sub_camera_name,
        patch_names=patch_names,
        reflectances=reflectances,
        grey_patch_name=grey_patch_name,
        shape=shape,
        noise_std=noise_std,
        rng=rng,
    )
    validation_main_rgb = _virtual_capture(
        validation_illuminants,
        camera_name=main_camera_name,
        patch_names=patch_names,
        reflectances=reflectances,
        grey_patch_name=grey_patch_name,
        shape=shape,
        noise_std=noise_std,
        rng=rng,
    )
    validation_sub_rgb = _virtual_capture(
        validation_illuminants,
        camera_name=sub_camera_name,
        patch_names=patch_names,
        reflectances=reflectances,
        grey_patch_name=grey_patch_name,
        shape=shape,
        noise_std=noise_std,
        rng=rng,
    )
    return SpectralDataset(
        train_illuminants=train_illuminants,
        validation_illuminants=validation_illuminants,
        patch_names=patch_names,
        train_main_white_points=_white_points(
            train_main_rgb,
            patch_names=patch_names,
            grey_patch_name=grey_patch_name,
        ),
        train_sub_white_points=_white_points(
            train_sub_rgb,
            patch_names=patch_names,
            grey_patch_name=grey_patch_name,
        ),
        train_main_rgb=train_main_rgb,
        train_sub_rgb=train_sub_rgb,
        validation_main_white_points=_white_points(
            validation_main_rgb,
            patch_names=patch_names,
            grey_patch_name=grey_patch_name,
        ),
        validation_sub_white_points=_white_points(
            validation_sub_rgb,
            patch_names=patch_names,
            grey_patch_name=grey_patch_name,
        ),
        validation_main_rgb=validation_main_rgb,
        validation_sub_rgb=validation_sub_rgb,
        main_ccm=_main_ccm(),
    )


def run_spectral_simulation(
    *,
    noise_std: float = 0.0005,
    seed: int = 13,
    l2: float = 1e-6,
    awb_degree: int = 2,
    ccm_degree: int = 2,
) -> SpectralSimulationReport:
    dataset = generate_spectral_dataset(noise_std=noise_std, seed=seed)
    awb_model = AWBSyncModel.fit(
        dataset.train_main_white_points,
        dataset.train_sub_white_points,
        degree=awb_degree,
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
        degree=ccm_degree,
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
            regularization=1e-6,
        )
        condition_numbers.append(float(np.linalg.cond(result.normalized_mapping @ result.sub_awb)))
        main_rgb = normalize_rgb(dataset.validation_main_rgb[idx])
        sub_rgb = normalize_rgb(dataset.validation_sub_rgb[idx])
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

    return SpectralSimulationReport(
        colorchecker=COLORCHECKER_NAME,
        main_camera=MAIN_CAMERA_NAME,
        sub_camera=SUB_CAMERA_NAME,
        wavelength_start_nm=int(SPECTRAL_SHAPE.start),
        wavelength_end_nm=int(SPECTRAL_SHAPE.end),
        wavelength_interval_nm=int(SPECTRAL_SHAPE.interval),
        train_illuminants=len(dataset.train_illuminants),
        validation_illuminants=len(dataset.validation_illuminants),
        patches_per_illuminant=len(dataset.patch_names),
        noise_std=float(noise_std),
        awb_naive_mean_error=float(np.mean(naive_awb_errors)),
        awb_sync_mean_error=float(np.mean(awb_errors)),
        awb_sync_p95_error=float(np.percentile(awb_errors, 95)),
        chroma_rmse_no_sync=normalized_rgb_rmse(no_sync_stack, main_stack),
        chroma_rmse_awb_only=normalized_rgb_rmse(awb_only_stack, main_stack),
        chroma_rmse_synced=normalized_rgb_rmse(synced_stack, main_stack),
        color_rmse_no_sync=relative_rgb_rmse(no_sync_stack, main_stack),
        color_rmse_awb_only=color_rmse_awb_only,
        color_rmse_synced=color_rmse_synced,
        color_improvement_vs_awb_only=float(color_rmse_awb_only / max(color_rmse_synced, 1e-12)),
        max_condition_number=float(np.max(condition_numbers)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AWB/CCM sync simulation from public spectral data.")
    parser.add_argument("--noise-std", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--awb-degree", type=int, default=2)
    parser.add_argument("--ccm-degree", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_spectral_simulation(
        noise_std=args.noise_std,
        seed=args.seed,
        l2=args.l2,
        awb_degree=args.awb_degree,
        ccm_degree=args.ccm_degree,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
