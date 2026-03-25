from __future__ import annotations

from typing import Any, Dict

from physics.calc_geom_metadata import FLOW_DIR, compute_geom_info

from .context import SampleContext

GEOMETRY_METRIC_FIELDNAMES = [
    "volume",
    "equivalent_diameter",
    "reference_area",
    "sphericity",
    "phi_cross",
]


def compute_geometry_metrics(context: SampleContext) -> Dict[str, Any]:
    cache_key = "geometry_metrics"

    if cache_key not in context.cache:
        try:
            res = compute_geom_info(context.stl_path, FLOW_DIR)

            context.cache[cache_key] = {
                "volume": res["Volume"],
                "equivalent_diameter": res["D_eq"],
                "reference_area": res["Reference_area"],
                "sphericity": res["Sphericity"],
                "phi_cross": res["Phi_Cross"],
            }

        except Exception as exc:
            print(f"[Warn] cannot compute geom info. Error: {exc}")
            # Fill with NaNs if the computation fails
            context.cache[cache_key] = {
                field: float("nan") for field in GEOMETRY_METRIC_FIELDNAMES
            }

    return context.cache[cache_key]
