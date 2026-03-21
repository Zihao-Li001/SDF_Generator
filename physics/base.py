from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol


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


class PhysicalVariableCalculator(Protocol):
    """Protocol for pluggable per-sample physical variable calculators."""

    name: str

    def fieldnames(self) -> List[str]:
        ...

    def compute(self, context: SampleContext) -> Mapping[str, Any]:
        ...
