from __future__ import annotations

from typing import Any, Dict

from physics.drag_coeff import (
    calculate_ke_drag,
    calculate_holzer_sommerfeld_drag,
)

from .geometry_metrics import compute_geometry_metrics
from .context import SampleContext

DRAG_FIELDNAMES = [
    "Cd_ke",
    "Cd_hs",
]


def compute_drag_outputs(context: SampleContext) -> Dict[str, Any]:
    """
    Compute the drag coefficient with different models

    Citation:
    cd_ke: https://doi.org/10.1016/j.powtec.2017.10.049
    cd_hs: https://doi.org/10.1016/j.powtec.2007.08.021
    """
    # 1. retrieve the geometry metrics
    geom = compute_geometry_metrics(context)

    try:
        # 2. compute Ke correlation
        cd_ke = calculate_ke_drag(
            context.reynolds_number,
            context.aspect_ratio,
            context.incident_angle,
        )

        # 3. compute Holzer-Sommerfeld correlation
        cd_hs = calculate_holzer_sommerfeld_drag(
            re=context.reynolds_number,
            phi=geom["sphericity"],
            phi_cross=geom["phi_cross"],
        )
    except Exception as exc:
        print(f"[Warn] Cd compute failed: {exc}")
        cd_ke = float("nan")
        cd_hs = float("nan")

    return {
        "Cd_ke": cd_ke,
        "Cd_hs": cd_hs,
    }
