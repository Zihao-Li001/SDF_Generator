from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from drag_coeff import calculate_drag_coefficient
from representation.calc_geom_metadata import FLOW_DIR, compute_geom_info

from .base import PhysicalVariableCalculator, SampleContext


@dataclass(frozen=True)
class DragCoefficientCalculator:
    name: str = "drag_coefficient"

    def fieldnames(self) -> List[str]:
        return ["Cd_equation"]

    def compute(self, context: SampleContext) -> Mapping[str, Any]:
        cd_eq = calculate_drag_coefficient(
            context.reynolds_number,
            context.aspect_ratio,
            context.incident_angle,
        )
        return {"Cd_equation": cd_eq}


@dataclass(frozen=True)
class GeometryMetricsCalculator:
    name: str = "geometry_metrics"

    def fieldnames(self) -> List[str]:
        return ["lRef", "Aref"]

    def compute(self, context: SampleContext) -> Mapping[str, Any]:
        cache_key = "geometry_metrics"
        if cache_key not in context.cache:
            try:
                _, d_eq, a_ref = compute_geom_info(context.stl_path, FLOW_DIR)
            except Exception as exc:
                print(f"[Warn] cannot compute geom info. Error: {exc}")
                d_eq, a_ref = float("nan"), float("nan")
            context.cache[cache_key] = {"lRef": d_eq, "Aref": a_ref}

        return context.cache[cache_key]


class CalculatorRegistry:
    def __init__(self, calculators: List[PhysicalVariableCalculator]):
        self.calculators = calculators

    def fieldnames(self) -> List[str]:
        names: List[str] = []
        for calculator in self.calculators:
            names.extend(calculator.fieldnames())
        return names

    def compute_all(self, context: SampleContext) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for calculator in self.calculators:
            overlap = set(outputs).intersection(calculator.fieldnames())
            if overlap:
                raise ValueError(
                    f"Calculator '{calculator.name}' overlaps existing fields: {sorted(overlap)}"
                )
            outputs.update(calculator.compute(context))
        return outputs


def build_default_registry() -> CalculatorRegistry:
    return CalculatorRegistry(
        calculators=[
            GeometryMetricsCalculator(),
            DragCoefficientCalculator(),
        ]
    )
