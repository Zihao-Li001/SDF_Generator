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

# ------ User setting
# Path to metadata.csv
META_CSV = Path(f'{CONFIG.OUTPUT["dataset_dir"]}/{CONFIG.OUTPUT["metadata_dir"]}')
# Path to STL file
STL_PATH_COLUME = "stl_file"
# Flow direction x-axis
FLOW_DIR = np.array([1.0, 0.0, 0.0])


def normalize(v):
    return v / np.linalg.norm(v)


def projected_area(mesh: trimesh.Trimesh, flow_dir: np.ndarray) -> float:
    """
    Compute projected area along flow direction
    """
    flow_dir = normalize(flow_dir)
    normals = mesh.face_normals
    areas = mesh.area_faces
    proj = np.abs(normals @ flow_dir) * areas
    return proj.sum()


def compute_geom_info(stl_path: Path, flow_dir: np.ndarray):
    """
    Compute volume, equivalent diameter, and projected area
    """
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")
    mesh = trimesh.load_mesh(stl_path, force="mesh")

    if not mesh.is_watertight:
        print(f"[Warn] {stl_path.name} not watertight, approx.")

    volume = mesh.volume
    d_eq = (6.0 * volume / np.pi) ** (1.0 / 3.0)
    a_proj = projected_area(mesh, flow_dir)
    return volume, d_eq, a_proj


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
