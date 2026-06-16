import numpy as np

from awb_ccm_sync import (
    AWBSyncModel,
    CCMSyncModel,
    awb_matrix_from_white_point,
    fit_local_ccm_matrix,
    run_spectral_simulation,
    run_synthetic_simulation,
    solve_ccm2,
)


def _white_point_grid() -> np.ndarray:
    u = np.linspace(0.65, 1.45, 4)
    v = np.linspace(0.55, 1.25, 4)
    return np.array([[ui, vi] for ui in u for vi in v], dtype=float)


def test_awb_sync_recovers_quadratic_white_point_mapping() -> None:
    main = _white_point_grid()
    sub = np.column_stack(
        [
            0.18 + 0.92 * main[:, 0] + 0.04 * main[:, 1] + 0.03 * main[:, 0] * main[:, 1],
            0.11 + 0.06 * main[:, 0] + 0.87 * main[:, 1] + 0.02 * main[:, 1] ** 2,
        ]
    )

    model = AWBSyncModel.fit(main, sub, degree=2, l2=1e-10)
    predicted = model.predict(main)

    np.testing.assert_allclose(predicted, sub, atol=1e-8)
    np.testing.assert_allclose(model.awb_matrix(main[0]), awb_matrix_from_white_point(sub[0]))


def test_local_ccm_matrix_recovers_two_g_normalized_linear_models() -> None:
    u = np.linspace(0.45, 1.65, 6)
    v = np.linspace(0.35, 1.45, 4)
    main_rgb = np.array([[ui, 1.0, vi] for ui in u for vi in v], dtype=float)
    expected = np.array(
        [
            [0.93, 0.0, 0.05],
            [0.07, 1.0, -0.02],
            [0.04, 0.0, 0.97],
        ]
    )
    sub_rgb = main_rgb @ expected

    actual = fit_local_ccm_matrix(main_rgb, sub_rgb)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_ccm_sync_recovers_white_point_conditioned_matrix() -> None:
    points = _white_point_grid()
    matrices = []
    for u, v in points:
        matrices.append(
            [
                [0.9 + 0.03 * u, 0.0, 0.02 + 0.01 * v],
                [0.05 - 0.02 * v, 1.0, -0.03 + 0.01 * u],
                [0.04 + 0.02 * u, 0.0, 0.95 - 0.03 * v],
            ]
        )
    matrices = np.asarray(matrices)

    model = CCMSyncModel.fit(points, matrices, degree=1, l2=1e-10)
    test_point = np.array([1.1, 0.8])
    expected = np.array(
        [
            [0.9 + 0.03 * test_point[0], 0.0, 0.02 + 0.01 * test_point[1]],
            [0.05 - 0.02 * test_point[1], 1.0, -0.03 + 0.01 * test_point[0]],
            [0.04 + 0.02 * test_point[0], 0.0, 0.95 - 0.03 * test_point[1]],
        ]
    )

    np.testing.assert_allclose(model.predict(test_point), expected, atol=1e-8)


def test_solve_ccm2_satisfies_documented_row_vector_equation() -> None:
    main_awb = awb_matrix_from_white_point([1.28, 0.74])
    sub_awb = awb_matrix_from_white_point([1.11, 0.91])
    main_ccm = np.array(
        [
            [1.2, -0.1, -0.1],
            [-0.05, 1.1, -0.05],
            [-0.02, -0.12, 1.14],
        ]
    )
    mapping = np.array(
        [
            [0.94, 0.0, 0.03],
            [0.06, 1.0, -0.01],
            [0.02, 0.0, 0.98],
        ]
    )

    sub_ccm = solve_ccm2(main_awb, main_ccm, sub_awb, mapping)

    np.testing.assert_allclose(mapping @ sub_awb @ sub_ccm, main_awb @ main_ccm, atol=1e-10)


def test_solve_ccm2_uses_fallback_when_matrix_is_ill_conditioned() -> None:
    fallback = np.eye(3)
    actual = solve_ccm2(
        np.eye(3),
        np.eye(3),
        np.eye(3),
        np.diag([1.0, 1e-9, 1.0]),
        cond_max=1e3,
        fallback_ccm=fallback,
    )

    np.testing.assert_allclose(actual, fallback)


def test_synthetic_simulation_validates_end_to_end_sync_path() -> None:
    report = run_synthetic_simulation(noise_std=0.0, seed=11, l2=1e-10)

    assert report.awb_sync_mean_error < 1e-8
    assert report.color_rmse_synced < 1e-8
    assert report.color_rmse_awb_only > 0.01
    assert report.max_condition_number < 10.0


def test_noisy_synthetic_simulation_keeps_sync_effective() -> None:
    report = run_synthetic_simulation(noise_std=0.001, seed=11, l2=1e-8)

    assert report.awb_sync_mean_error < report.awb_naive_mean_error * 0.01
    assert report.color_rmse_synced < 0.01
    assert report.color_rmse_synced < report.color_rmse_awb_only * 0.25


def test_public_spectral_simulation_improves_holdout_illuminants() -> None:
    report = run_spectral_simulation(noise_std=0.0005, seed=13, l2=1e-6)

    assert report.train_illuminants == 51
    assert report.validation_illuminants == 8
    assert report.awb_sync_mean_error < report.awb_naive_mean_error * 0.05
    assert report.chroma_rmse_synced < report.chroma_rmse_awb_only * 0.5
    assert report.color_rmse_synced < report.color_rmse_awb_only * 0.5
    assert report.max_condition_number < 50.0
