# data/utils/calc_geom_metadata.py
"""
Batch compute particle volume, volume-equivalent diameter, and project area
for STL geometries
Updata metadata.csv accordingly

Author: Li's pipeline of geom generation
Dependencies: trimesh, pandas, numpy
"""

import trimesh
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.errors import TopologicalError
from shapely.geometry import Polygon
from shapely.ops import unary_union
from config import CONFIG

# --- User setting --- #
# Path to metadata.csv
META_CSV = Path(f'{CONFIG.OUTPUT["dataset_dir"]}/{CONFIG.OUTPUT["metadata_dir"]}')
# Path to STL file
STL_PATH_COLUME = "stl_file"
# Flow direction x-axis
FLOW_DIR = np.array([1.0, 0.0, 0.0])


def normalize(v):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("projection direction must be finite and non-zero")
    return v / norm


def _projection_basis(direction: np.ndarray):
    """Return a stable orthonormal basis for the plane normal to direction."""
    direction = normalize(np.asarray(direction, dtype=float))
    seed = np.eye(3)[np.argmin(np.abs(direction))]
    axis_u = normalize(np.cross(direction, seed))
    axis_v = np.cross(direction, axis_u)
    return axis_u, axis_v


def _union_projected_triangles(projected: np.ndarray, scale: float) -> float:
    """Union projected triangles, retrying once on a scale-aware precision grid."""
    area_tolerance = 64.0 * np.finfo(float).eps * scale**2
    chunk_size = 512
    last_error = None

    for grid_size in (None, 1.0e-12 * scale):
        coords = projected
        if grid_size is not None:
            coords = np.round(coords / grid_size) * grid_size

        edges_1 = coords[:, 1] - coords[:, 0]
        edges_2 = coords[:, 2] - coords[:, 0]
        twice_areas = np.abs(
            edges_1[:, 0] * edges_2[:, 1]
            - edges_1[:, 1] * edges_2[:, 0]
        )
        valid = twice_areas > 2.0 * area_tolerance
        if not np.any(valid):
            raise ValueError("mesh has no non-degenerate projected triangles")

        polygons = [Polygon(triangle) for triangle in coords[valid]]
        try:
            partial_unions = [
                unary_union(polygons[start : start + chunk_size])
                for start in range(0, len(polygons), chunk_size)
            ]
            geometry = unary_union(partial_unions)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            area = float(geometry.area)
            if geometry.is_valid and np.isfinite(area) and area > 0.0:
                return area
            last_error = ValueError("projected polygon union is invalid or empty")
        except (TopologicalError, ValueError) as exc:
            last_error = exc

    raise ValueError(f"cannot construct projected triangle union: {last_error}")


def projected_area(mesh: trimesh.Trimesh, direction: np.ndarray) -> float:
    """
    Compute the true silhouette area normal to ``direction``.

    Every mesh triangle is orthogonally projected to a two-dimensional
    basis and the returned area is the geometric union of those projected
    triangles.  This avoids double-counting occluded or overlapping surface
    patches on non-convex particles.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("mesh is empty")

    triangles = np.asarray(mesh.triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("mesh does not contain triangular faces")
    if not np.all(np.isfinite(triangles)):
        raise ValueError("mesh contains non-finite coordinates")

    axis_u, axis_v = _projection_basis(direction)
    projected = np.stack(
        (triangles @ axis_u, triangles @ axis_v),
        axis=-1,
    )
    spans = np.ptp(projected.reshape(-1, 2), axis=0)
    scale = float(np.max(spans))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mesh projection has zero or invalid extent")

    return _union_projected_triangles(projected, scale)


def compute_sphericity(volume: float, surface_area: float) -> float:
    """
    Compute Wadell's true sphericity.
    Ratio of the surface area of a volume-equivalent sphere
    to the particle's surface area.
    """
    if surface_area == 0:
        return 0.0
    return (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_area


def compute_geom_info(stl_path: Path, flow_dir: np.ndarray):
    mesh = trimesh.load_mesh(stl_path, force="mesh")

    volume = mesh.volume
    surface_area = mesh.area
    d_eq = (6.0 * volume / np.pi) ** (1.0 / 3.0)

    # 1. Area of the volume-equivalent sphere cross-section
    a_sphere_cs = (np.pi / 4.0) * (d_eq**2)

    # 2. Crosswise Projected Area (Facing the flow)
    a_proj_cross = projected_area(mesh, flow_dir)

    # --- Calculations ---
    # Standard Wadell Sphericity
    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area

    # Hölzer & Sommerfeld Sphericity
    phi_cross = a_sphere_cs / a_proj_cross

    return {
        "Volume": volume,
        "D_eq": d_eq,
        "Reference_area": a_proj_cross,
        "Sphericity": sphericity,
        "Phi_Cross": phi_cross,
    }


def main():
    if not META_CSV.exists():
        raise FileNotFoundError(f"metadata file not found: {META_CSV}")

    df = pd.read_csv(META_CSV)

    if STL_PATH_COLUME not in df.columns:
        raise KeyError(f"Column '{STL_PATH_COLUME}' not found in {META_CSV}")

    volumes, d_eqs, a_projs = [], [], []

    for i, row in df.iterrows():
        stl_path = Path(row[STL_PATH_COLUME])
        try:
            V, Dv, Ap = compute_geom_info(stl_path, FLOW_DIR)
        except Exception as e:
            print(f"[Error] {stl_path}: {e}")
            V, Dv, Ap = np.nan, np.nan, np.nan

        volumes.append(V)
        d_eqs.append(Dv)
        a_projs.append(Ap)

    df["Volume"] = volumes
    df["D_eq"] = d_eqs
    df["A_proj"] = a_projs

    df.to_csv(META_CSV, index=False)
    print("metadata updata")


if __name__ == "__main__":
    main()
