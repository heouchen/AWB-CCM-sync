"""Reference implementation for cross-module AWB/CCM synchronization."""

from .models import (
    AWBSyncModel,
    CCMSyncModel,
    SyncResult,
    awb_matrix_from_white_point,
    fit_local_ccm_matrix,
    normalize_rgb,
    solve_ccm2,
    sync_runtime,
)
from .polynomial import Polynomial2D, polynomial_terms
from .simulation import SimulationReport, run_synthetic_simulation
from .spectral_simulation import SpectralSimulationReport, run_spectral_simulation

__all__ = [
    "AWBSyncModel",
    "CCMSyncModel",
    "Polynomial2D",
    "SyncResult",
    "awb_matrix_from_white_point",
    "fit_local_ccm_matrix",
    "normalize_rgb",
    "polynomial_terms",
    "run_spectral_simulation",
    "run_synthetic_simulation",
    "solve_ccm2",
    "SimulationReport",
    "SpectralSimulationReport",
    "sync_runtime",
]
