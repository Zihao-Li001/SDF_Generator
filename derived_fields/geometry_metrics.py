from __future__ import annotations

from typing import Any, Dict

from representation.calc_geom_metadata import FLOW_DIR, compute_geom_info

from .context import SampleContext

GEOMETRY_METRIC_FIELDNAMES = ["lRef", "Aref"]


def compute_geometry_metrics(context: SampleContext) -> Dict[str, Any]:
    cache_key = "geometry_metrics"
    if cache_key not in context.cache:
        try:
            _, d_eq, a_ref = compute_geom_info(context.stl_path, FLOW_DIR)
        except Exception as exc:
            print(f"[Warn] cannot compute geom info. Error: {exc}")
            d_eq, a_ref = float("nan"), float("nan")
        context.cache[cache_key] = {"lRef": d_eq, "Aref": a_ref}

    return context.cache[cache_key]
