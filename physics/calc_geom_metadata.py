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
from config import CONFIG

# --- User setting --- #
# Path to metadata.csv
META_CSV = Path(f'{CONFIG.OUTPUT["dataset_dir"]}/{CONFIG.OUTPUT["metadata_dir"]}')
# Path to STL file
STL_PATH_COLUME = "stl_file"
# Flow direction x-axis
FLOW_DIR = np.array([1.0, 0.0, 0.0])


def normalize(v):
    return v / np.linalg.norm(v)


def projected_area(mesh: trimesh.Trimesh, direction: np.ndarray) -> float:
    """
    Compute the actual silhouette area along a specific direction.
    Note: sum(abs(normals @ dir) * face_areas) returns TWICE the
    projected silhouette area for a closed, watertight mesh.
    """
    direction = normalize(direction)
    normals = mesh.face_normals
    areas = mesh.area_faces
    # The dot product method integrated over the surface gives 2 * A_proj
    total_proj = np.abs(normals @ direction) * areas
    return 0.5 * total_proj.sum()


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
