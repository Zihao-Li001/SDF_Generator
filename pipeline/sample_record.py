from __future__ import annotations

from typing import Any, Dict

from derived_fields.context import SampleContext

BASE_METADATA_FIELDS = [
    "sample_id",
    "geom_id",
    "rotate_id",
    "aspect_ratio",
    "incident_angle",
    "Re",
    "stl_path",
    "voxel_path",
    "sdf_path",
]


def build_metadata_record(
    context: SampleContext,
    derived_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    combine the context and derived-output togother as one CSV row
    """
    return {
        "sample_id": context.sample_id,
        "geom_id": context.geom_id,
        "rotate_id": context.rotate_id,
        "aspect_ratio": context.aspect_ratio,
        "incident_angle": context.incident_angle,
        "Re": context.reynolds_number,
        **derived_outputs,
        "stl_path": context.stl_path.relative_to(context.dataset_root).as_posix(),
        "voxel_path": context.voxel_path.relative_to(context.dataset_root).as_posix(),
        "sdf_path": context.sdf_path.relative_to(context.dataset_root).as_posix(),
    }
