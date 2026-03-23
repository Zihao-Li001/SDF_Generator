from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class SampleContext:
    sample_id: int
    geom_id: int
    rotate_id: int
    aspect_ratio: float
    d2: float
    d9: float
    incident_angle: float
    reynolds_number: float
    dataset_root: Path
    stl_path: Path
    voxel_path: Path
    sdf_path: Path
    cache: Dict[str, Any] = field(default_factory=dict)


def build_sample_context(
    dataset_root: Path,
    geom_id: int,
    rotate_id: int,
    aspect_ratio: float,
    d2: float,
    d9: float,
    incident_angle: float,
    reynolds_number: float,
    stl_path: Path,
    voxel_path: Path,
    sdf_path: Path,
) -> SampleContext:
    return SampleContext(
        sample_id=geom_id * 1000 + rotate_id,
        geom_id=geom_id,
        rotate_id=rotate_id,
        aspect_ratio=aspect_ratio,
        d2=d2,
        d9=d9,
        incident_angle=incident_angle,
        reynolds_number=reynolds_number,
        dataset_root=dataset_root,
        stl_path=stl_path,
        voxel_path=voxel_path,
        sdf_path=sdf_path,
    )
