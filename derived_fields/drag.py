from __future__ import annotations

from typing import Any, Dict

from physics.drag_coeff import calculate_drag_coefficient

from .context import SampleContext

DRAG_FIELDNAMES = ["Cd_ke"]


def compute_drag_outputs(context: SampleContext) -> Dict[str, Any]:
    """
    Compute the drag coefficient
    Ref:
        Chunhai Ke, On the drag coefficient and averaged Nusselt number of an
        ellipsoidal particle in a fluid.
    """
    cd_ke = calculate_drag_coefficient(
        context.reynolds_number,
        context.aspect_ratio,
        context.incident_angle,
    )
    return {"Cd_ke": cd_ke}
