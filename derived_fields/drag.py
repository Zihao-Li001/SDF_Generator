from __future__ import annotations

from typing import Any, Dict

from drag_coeff import calculate_drag_coefficient

from .context import SampleContext

DRAG_FIELDNAMES = ["Cd_equation"]


def compute_drag_outputs(context: SampleContext) -> Dict[str, Any]:
    cd_eq = calculate_drag_coefficient(
        context.reynolds_number,
        context.aspect_ratio,
        context.incident_angle,
    )
    return {"Cd_equation": cd_eq}
