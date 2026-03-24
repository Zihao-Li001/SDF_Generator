from __future__ import annotations

from typing import Any, Dict

from physics.calc_geom_metadata import FLOW_DIR, compute_geom_info

from .context import SampleContext

GEOMETRY_METRIC_FIELDNAMES = [
    "volume",
    "equivalent_diameter",
    "reference_area",
    "sphericity",
]


def compute_geometry_metrics(context: SampleContext) -> Dict[str, Any]:
    cache_key = "geometry_metrics"
    if cache_key not in context.cache:
        try:
            v, d_eq, a_ref, s = compute_geom_info(context.stl_path, FLOW_DIR)
        except Exception as exc:
            print(f"[Warn] cannot compute geom info. Error: {exc}")
            v, d_eq, a_ref, s = float("nan"), float("nan"), float("nan"), float("nan")
        context.cache[cache_key] = {
            "volume": v,
            "equivalent_diameter": d_eq,
            "reference_area": a_ref,
            "sphericity": s,
        }

    return context.cache[cache_key]
