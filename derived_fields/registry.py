from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .context import SampleContext
from .drag import DRAG_FIELDNAMES, compute_drag_outputs
from .geometry_metrics import GEOMETRY_METRIC_FIELDNAMES, compute_geometry_metrics

DerivedFieldProvider = Callable[[SampleContext], Dict[str, Any]]
DerivedFieldEntry = Tuple[Sequence[str], DerivedFieldProvider]

DEFAULT_DERIVED_FIELD_PROVIDERS: List[DerivedFieldEntry] = [
    (GEOMETRY_METRIC_FIELDNAMES, compute_geometry_metrics),
    (DRAG_FIELDNAMES, compute_drag_outputs),
]


def get_default_derived_field_providers() -> List[DerivedFieldEntry]:
    return list(DEFAULT_DERIVED_FIELD_PROVIDERS)


def get_derived_fieldnames(
    providers: Iterable[DerivedFieldEntry] | None = None,
) -> List[str]:
    active_providers = (
        DEFAULT_DERIVED_FIELD_PROVIDERS if providers is None else list(providers)
    )
    fieldnames: List[str] = []
    for names, _ in active_providers:
        fieldnames.extend(names)
    return fieldnames


def compute_derived_fields(
    context: SampleContext,
    providers: Iterable[DerivedFieldEntry] | None = None,
) -> Dict[str, Any]:
    active_providers = (
        DEFAULT_DERIVED_FIELD_PROVIDERS if providers is None else list(providers)
    )
    outputs: Dict[str, Any] = {}
    for fieldnames, provider in active_providers:
        overlap = set(outputs).intersection(fieldnames)
        if overlap:
            raise ValueError(
                f"Derived field overlap detected for {provider.__name__}: {sorted(overlap)}"
            )
        outputs.update(provider(context))
    return outputs
