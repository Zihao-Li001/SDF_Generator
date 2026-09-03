#!/usr/bin/env python3
"""Audit and update crosswise sphericity from stored STL silhouettes.

The command is dry-run by default.  It updates a metadata file only when every
row in that file passes mapping, mesh, projection, and validation checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh

from physics.calc_geom_metadata import normalize, projected_area


REQUIRED_COLUMNS = {
    "sample_id",
    "geom_id",
    "rotate_id",
    "incident_angle",
    "volume",
    "equivalent_diameter",
    "reference_area",
    "sphericity",
    "phi_cross",
    "stl_path",
}
SOURCE_COMPARE_COLUMNS = (
    "rotate_id",
    "aspect_ratio",
    "incident_angle",
    "Re",
    "d2",
    "d9",
    "volume",
    "equivalent_diameter",
    "reference_area",
    "sphericity",
)
AREA_RTOL = 1.0e-8
AREA_ATOL = 1.0e-10
CONVEX_RTOL = 1.0e-8


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _load_mesh_metrics(task: tuple[str, tuple[float, float, float]]) -> dict[str, Any]:
    path_text, direction_values = task
    path = Path(path_text)
    try:
        mesh = trimesh.load_mesh(path, force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"STL did not load as Trimesh: {type(mesh).__name__}")
        if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            raise ValueError("mesh is empty")
        if not np.all(np.isfinite(mesh.vertices)):
            raise ValueError("mesh contains non-finite vertices")

        direction = normalize(np.asarray(direction_values, dtype=float))
        old_area = float(
            0.5
            * np.sum(
                np.abs(np.asarray(mesh.face_normals) @ direction)
                * np.asarray(mesh.area_faces)
            )
        )
        union_area = float(projected_area(mesh, direction))
        volume = abs(float(mesh.volume))
        if not all(math.isfinite(x) and x > 0.0 for x in (volume, old_area, union_area)):
            raise ValueError("volume or projected area is not finite and positive")

        return {
            "resolved_stl": str(path),
            "compute_error": "",
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "volume_stl": volume,
            "old_area_stl": old_area,
            "union_area": union_area,
        }
    except Exception as exc:  # recorded per row; never converted to NaN silently
        return {
            "resolved_stl": str(path),
            "compute_error": f"{type(exc).__name__}: {exc}",
            "faces": None,
            "watertight": False,
            "volume_stl": None,
            "old_area_stl": None,
            "union_area": None,
        }


class MetadataResolver:
    def __init__(self, project_root: Path, rtol: float, atol: float):
        self.project_root = project_root
        self.rtol = rtol
        self.atol = atol
        self._source_tables: dict[Path, pd.DataFrame] = {}

    def _source_table(self, path: Path) -> pd.DataFrame:
        path = path.resolve()
        if path not in self._source_tables:
            table = pd.read_csv(path)
            if "sample_id" not in table.columns or table["sample_id"].duplicated().any():
                raise ValueError(f"source metadata has missing/duplicate sample_id: {path}")
            self._source_tables[path] = table.set_index("sample_id", drop=False)
        return self._source_tables[path]

    def _validate_source_row(
        self,
        current: pd.Series,
        source_metadata: Path,
        source_id: int,
    ) -> None:
        table = self._source_table(source_metadata)
        if source_id not in table.index:
            raise KeyError(f"source sample_id {source_id} not in {source_metadata}")
        source = table.loc[source_id]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"source sample_id {source_id} is not unique")

        for column in SOURCE_COMPARE_COLUMNS:
            if column not in current.index or column not in source.index:
                continue
            left = current[column]
            right = source[column]
            if pd.isna(left) and pd.isna(right):
                continue
            try:
                equal = np.isclose(
                    float(left),
                    float(right),
                    rtol=self.rtol,
                    atol=self.atol,
                )
            except (TypeError, ValueError):
                equal = str(left) == str(right)
            if not bool(equal):
                raise ValueError(
                    f"source metadata mismatch for {column}: {left!r} != {right!r}"
                )

    def resolve(
        self,
        dataset: dict[str, Any],
        metadata_path: Path,
        row: pd.Series,
    ) -> Path:
        sample_id = int(row["sample_id"])
        geom_id = int(row["geom_id"])
        rotate_id = int(row["rotate_id"])
        expected_sample_id = geom_id * 1000 + rotate_id
        if sample_id != expected_sample_id:
            raise ValueError(
                f"sample_id rule failed: {sample_id} != {expected_sample_id}"
            )

        if dataset.get("resolver", "direct") == "direct":
            raw_path = Path(str(row["stl_path"]))
            return (raw_path if raw_path.is_absolute() else metadata_path.parent / raw_path).resolve()

        d9 = _safe_float(row["d9"])
        for rule in dataset.get("stl_rules", []):
            if not np.isclose(d9, float(rule["d9"]), rtol=0.0, atol=1.0e-12):
                continue
            source_geom_id = geom_id + int(rule.get("geom_id_offset", 0))
            source_id = source_geom_id * 1000 + rotate_id
            source_metadata = (self.project_root / rule["source_metadata"]).resolve()
            self._validate_source_row(row, source_metadata, source_id)
            return (self.project_root / rule["stl_dir"] / f"{source_id}.stl").resolve()

        raise ValueError(f"no STL mapping rule for d9={d9}")


def _base_report_row(
    dataset_name: str,
    metadata_path: Path,
    row_index: int,
    row: pd.Series,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "metadata": str(metadata_path),
        "row_index": row_index,
        "sample_id": int(row["sample_id"]),
        "geom_id": int(row["geom_id"]),
        "rotate_id": int(row["rotate_id"]),
        "aspect_ratio": row.get("aspect_ratio", None),
        "incident_angle": row.get("incident_angle", None),
        "d2": row.get("d2", None),
        "d9": row.get("d9", None),
        "stl_path_metadata": row["stl_path"],
        "resolved_stl": "",
        "status": "pending",
        "error": "",
        "faces": None,
        "watertight": None,
        "volume_metadata": row["volume"],
        "volume_stl": None,
        "equivalent_diameter": row["equivalent_diameter"],
        "old_area_metadata": row["reference_area"],
        "old_area_stl": None,
        "union_area": None,
        "area_reduction": None,
        "old_phi": row["phi_cross"],
        "new_phi": None,
        "relative_change": None,
    }


def _patch_phi_column(
    original: bytes,
    replacements: dict[int, str],
    expected_rows: int,
) -> bytes:
    text = original.decode("utf-8")
    lines = text.splitlines(keepends=True)
    if len(lines) != expected_rows + 1:
        raise ValueError(
            f"physical CSV line count {len(lines) - 1} != expected {expected_rows}"
        )

    header_text = lines[0].rstrip("\r\n")
    if '"' in header_text:
        raise ValueError("quoted CSV headers are not supported by exact-text updater")
    header = header_text.split(",")
    if header.count("phi_cross") != 1:
        raise ValueError("metadata must contain exactly one phi_cross column")
    phi_index = header.index("phi_cross")

    output = [lines[0]]
    for row_index, line in enumerate(lines[1:]):
        if line.endswith("\r\n"):
            ending, body = "\r\n", line[:-2]
        elif line.endswith("\n") or line.endswith("\r"):
            ending, body = line[-1], line[:-1]
        else:
            ending, body = "", line
        if '"' in body:
            raise ValueError("quoted CSV rows are not supported by exact-text updater")
        fields = body.split(",")
        if len(fields) != len(header):
            raise ValueError(
                f"row {row_index + 2} has {len(fields)} fields; expected {len(header)}"
            )
        if row_index not in replacements:
            raise ValueError(f"missing phi_cross replacement for row {row_index}")
        fields[phi_index] = replacements[row_index]
        output.append(",".join(fields) + ending)

    candidate = "".join(output).encode("utf-8")
    _verify_only_phi_changed(original, candidate)
    return candidate


def _verify_only_phi_changed(original: bytes, candidate: bytes) -> None:
    old_lines = original.decode("utf-8").splitlines()
    new_lines = candidate.decode("utf-8").splitlines()
    if len(old_lines) != len(new_lines) or old_lines[0] != new_lines[0]:
        raise ValueError("candidate changed CSV row count or header")
    header = old_lines[0].split(",")
    phi_index = header.index("phi_cross")
    for line_number, (old_line, new_line) in enumerate(
        zip(old_lines[1:], new_lines[1:]), start=2
    ):
        old_fields = old_line.split(",")
        new_fields = new_line.split(",")
        if len(old_fields) != len(new_fields):
            raise ValueError(f"candidate changed field count on line {line_number}")
        for field_index, (old, new) in enumerate(zip(old_fields, new_fields)):
            if field_index != phi_index and old != new:
                raise ValueError(
                    f"candidate changed non-target field {header[field_index]} "
                    f"on line {line_number}"
                )


def _atomic_replace(path: Path, data: bytes) -> None:
    original_mode = path.stat().st_mode
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _select_validation_samples(rows: pd.DataFrame) -> pd.DataFrame:
    valid = rows[rows["status"] == "ok"].copy()
    if valid.empty:
        return valid
    selected: list[int] = []
    rng = random.Random(20260903)
    selected.extend(rng.sample(list(valid.index), min(8, len(valid))))
    selected.extend(valid.nlargest(min(8, len(valid)), "relative_change").index)
    selected.extend(valid.nsmallest(min(4, len(valid)), "sphericity_sort").index)
    if valid["irregularity_sort"].notna().any():
        selected.extend(valid.nlargest(min(4, len(valid)), "irregularity_sort").index)
    return valid.loc[list(dict.fromkeys(selected))]


def run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")

    project_root = (manifest_path.parent / manifest["project_root"]).resolve()
    direction = tuple(float(x) for x in manifest["flow_direction"])
    if len(direction) != 3:
        raise ValueError("flow_direction must have three components")
    mapping_rtol = float(manifest.get("mapping_rtol", 2.0e-7))
    mapping_atol = float(manifest.get("mapping_atol", 2.0e-9))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir
        else project_root / "data" / "crosswise_sphericity_audit" / timestamp
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir = report_dir / "candidates"
    candidate_dir.mkdir()

    resolver = MetadataResolver(project_root, mapping_rtol, mapping_atol)
    datasets: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    path_to_report_indices: dict[Path, list[int]] = {}

    for dataset in manifest["datasets"]:
        metadata_path = (project_root / dataset["metadata"]).resolve()
        original = metadata_path.read_bytes()
        frame = pd.read_csv(metadata_path)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{metadata_path} missing columns: {sorted(missing_columns)}")
        if len(frame) != int(dataset["expected_rows"]):
            raise ValueError(
                f"{metadata_path}: {len(frame)} rows != expected {dataset['expected_rows']}"
            )
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{metadata_path} contains duplicate sample_id")

        dataset_info = {
            **dataset,
            "metadata_path": metadata_path,
            "original": original,
            "original_sha256": _sha256_bytes(original),
            "row_report_indices": [],
        }
        datasets.append(dataset_info)

        for row_index, row in frame.iterrows():
            report = _base_report_row(dataset["name"], metadata_path, row_index, row)
            report["sphericity_sort"] = _safe_float(row["sphericity"])
            d2 = float(row["d2"]) if "d2" in row.index and pd.notna(row["d2"]) else 0.0
            d9 = float(row["d9"]) if "d9" in row.index and pd.notna(row["d9"]) else 0.0
            report["irregularity_sort"] = d2 + d9
            report_index = len(report_rows)
            report_rows.append(report)
            dataset_info["row_report_indices"].append(report_index)
            try:
                resolved = resolver.resolve(dataset, metadata_path, row)
                report["resolved_stl"] = str(resolved)
                if not resolved.is_file():
                    raise FileNotFoundError(f"ground-truth STL not found: {resolved}")
                path_to_report_indices.setdefault(resolved, []).append(report_index)
            except Exception as exc:
                report["status"] = "mapping_or_missing_error"
                report["error"] = f"{type(exc).__name__}: {exc}"

    tasks = [(str(path), direction) for path in path_to_report_indices]
    print(
        f"Resolved {len(tasks)} unique STL files for {len(report_rows)} metadata rows; "
        f"workers={args.workers}",
        flush=True,
    )
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for geometry in executor.map(_load_mesh_metrics, tasks, chunksize=8):
            resolved = Path(geometry["resolved_stl"]).resolve()
            for report_index in path_to_report_indices[resolved]:
                report_rows[report_index].update(geometry)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Computed {completed}/{len(tasks)} unique STL projections", flush=True)

    for dataset in datasets:
        expect_convex = bool(dataset.get("expect_convex_equality", False))
        for report_index in dataset["row_report_indices"]:
            report = report_rows[report_index]
            if report["status"] != "pending":
                continue
            if report.get("compute_error"):
                report["status"] = "projection_error"
                report["error"] = report["compute_error"]
                continue
            if not report["watertight"]:
                report["status"] = "invalid_mesh"
                report["error"] = "mesh is not watertight"
                continue

            volume_metadata = _safe_float(report["volume_metadata"])
            old_area_metadata = _safe_float(report["old_area_metadata"])
            if not np.isclose(
                report["volume_stl"], volume_metadata, rtol=mapping_rtol, atol=mapping_atol
            ):
                report["status"] = "geometry_identity_error"
                report["error"] = (
                    f"STL volume {report['volume_stl']:.17g} does not match "
                    f"metadata {volume_metadata:.17g}"
                )
                continue
            if not np.isclose(
                report["old_area_stl"],
                old_area_metadata,
                rtol=mapping_rtol,
                atol=mapping_atol,
            ):
                report["status"] = "geometry_identity_error"
                report["error"] = (
                    f"STL old area {report['old_area_stl']:.17g} does not match "
                    f"metadata {old_area_metadata:.17g}"
                )
                continue

            area_tolerance = max(AREA_ATOL, AREA_RTOL * report["old_area_stl"])
            if report["union_area"] > report["old_area_stl"] + area_tolerance:
                report["status"] = "area_inequality_error"
                report["error"] = "union area exceeds old surface-integral area"
                continue
            area_reduction = (
                report["old_area_stl"] - report["union_area"]
            ) / report["old_area_stl"]
            report["area_reduction"] = area_reduction
            if expect_convex and abs(area_reduction) > CONVEX_RTOL:
                report["status"] = "convex_equality_error"
                report["error"] = (
                    f"convex old/union relative difference {area_reduction:.3e} "
                    f"exceeds {CONVEX_RTOL:.1e}"
                )
                continue

            diameter = _safe_float(report["equivalent_diameter"])
            old_phi = _safe_float(report["old_phi"])
            new_phi = math.pi * diameter**2 / (4.0 * report["union_area"])
            if not math.isfinite(new_phi) or new_phi <= 0.0:
                report["status"] = "crosswise_sphericity_error"
                report["error"] = "new phi_cross is not finite and positive"
                continue
            report["new_phi"] = new_phi
            report["relative_change"] = abs(new_phi - old_phi) / abs(old_phi)
            report["status"] = "ok"

    rows = pd.DataFrame(report_rows)
    dataset_summaries: list[dict[str, Any]] = []
    candidates: dict[Path, bytes] = {}
    eligible_names: set[str] = set()
    for dataset in datasets:
        subset = rows.loc[dataset["row_report_indices"]]
        failures = subset[subset["status"] != "ok"]
        eligible = failures.empty and len(subset) == int(dataset["expected_rows"])
        if eligible:
            replacements = {
                int(row["row_index"]): format(float(row["new_phi"]), ".17g")
                for _, row in subset.iterrows()
            }
            candidate = _patch_phi_column(
                dataset["original"], replacements, int(dataset["expected_rows"])
            )
            candidate_path = candidate_dir / f"{dataset['name']}.metadata.csv"
            candidate_path.write_bytes(candidate)
            candidates[dataset["metadata_path"]] = candidate
            eligible_names.add(dataset["name"])
        dataset_summaries.append(
            {
                "dataset": dataset["name"],
                "metadata": str(dataset["metadata_path"]),
                "rows": len(subset),
                "computed_ok": int((subset["status"] == "ok").sum()),
                "failed_or_missing": int((subset["status"] != "ok").sum()),
                "eligible_for_atomic_update": eligible,
                "status_counts": subset["status"].value_counts().to_dict(),
            }
        )

    valid_rows = rows[rows["status"] == "ok"]
    skipped_atomic = int(
        sum(
            1
            for report in report_rows
            if report["status"] == "ok" and report["dataset"] not in eligible_names
        )
    )
    backup_manifest: list[dict[str, Any]] = []
    updated_rows = 0

    if args.apply:
        for dataset in datasets:
            path = dataset["metadata_path"]
            if dataset["name"] not in eligible_names:
                continue
            if _sha256_file(path) != dataset["original_sha256"]:
                raise RuntimeError(f"metadata changed during computation: {path}")
            backup = path.with_name(f"{path.name}.pre_crosswise_union.{timestamp}.bak")
            if backup.exists():
                raise FileExistsError(f"backup path already exists: {backup}")
            shutil.copy2(path, backup)
            backup_sha = _sha256_file(backup)
            if backup_sha != dataset["original_sha256"]:
                raise RuntimeError(f"backup verification failed: {backup}")
            backup_manifest.append(
                {
                    "dataset": dataset["name"],
                    "metadata": str(path),
                    "backup": str(backup),
                    "original_sha256": dataset["original_sha256"],
                    "backup_sha256": backup_sha,
                }
            )

        for dataset in datasets:
            path = dataset["metadata_path"]
            if dataset["name"] not in eligible_names:
                continue
            _atomic_replace(path, candidates[path])
            _verify_only_phi_changed(dataset["original"], path.read_bytes())
            updated_rows += int(dataset["expected_rows"])

    rows.drop(columns=["sphericity_sort", "irregularity_sort"], errors="ignore").to_csv(
        report_dir / "rows.csv", index=False
    )
    validation_samples = _select_validation_samples(rows)
    validation_samples.drop(
        columns=["sphericity_sort", "irregularity_sort"], errors="ignore"
    ).to_csv(report_dir / "validation_samples.csv", index=False)
    with (report_dir / "backup_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(backup_manifest), stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    relative_changes = valid_rows["relative_change"].astype(float)
    largest = valid_rows.nlargest(min(10, len(valid_rows)), "relative_change")
    summary = {
        "timestamp": timestamp,
        "mode": "apply" if args.apply else "dry-run",
        "manifest": str(manifest_path),
        "project_root": str(project_root),
        "flow_direction": list(direction),
        "scope_files": len(datasets),
        "scope_rows": len(rows),
        "unique_stl_computed": len(tasks),
        "successfully_computed_rows": len(valid_rows),
        "failed_or_missing_rows": int((rows["status"] != "ok").sum()),
        "eligible_files": len(eligible_names),
        "would_update_rows": int(
            sum(item["rows"] for item in dataset_summaries if item["eligible_for_atomic_update"])
        ),
        "successfully_updated_rows": updated_rows,
        "skipped_due_file_atomic_rows": skipped_atomic,
        "relative_change": {
            "mean": float(relative_changes.mean()),
            "median": float(relative_changes.median()),
            "max": float(relative_changes.max()),
        },
        "area_inequality_violations": int(
            (rows["status"] == "area_inequality_error").sum()
        ),
        "largest_relative_changes": [
            {
                "dataset": row["dataset"],
                "sample_id": int(row["sample_id"]),
                "old_phi": float(row["old_phi"]),
                "new_phi": float(row["new_phi"]),
                "relative_change": float(row["relative_change"]),
                "old_area": float(row["old_area_stl"]),
                "union_area": float(row["union_area"]),
            }
            for _, row in largest.iterrows()
        ],
        "datasets": dataset_summaries,
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(summary), stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), flush=True)
    print(f"Audit report: {report_dir}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and atomically replace only fully validated metadata files.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
