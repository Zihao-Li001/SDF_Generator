# representation/geom_generator.py
"""
High-level coordinator for generating particles with Spherical Harmonics.
Uses funcs.py for geometry and SHPSG.py for coefficients.
"""
import numpy as np
from .SHPSG import SHPSG
from . import funcs


def create_base_sh_mesh(level=2):
    """
    Creates and refines a base mesh for SHPSG.
    """
    current_vertices, current_faces = funcs.icosahedron()

    for i in range(level):
        current_vertices, current_faces = funcs.subdivided_mesh(
            current_vertices, current_faces
        )

    sph_coords = funcs.car2sph(current_vertices)

    print("Create of Base Mesh Finish")
    return {
        "vertices": current_vertices,
        "faces": current_faces,
        "sph_coords": sph_coords,
    }


def generate_sh_particle(
    Ar,
    d2,
    d9,
    base_mesh_data,
    incident_angle=0.0,
):
    """
    Generates a single SH particle's geometry.
    1. Gets coefficients from SHPSG.
    2. Calculates final vertices using funcs.
    """
    coeffs = SHPSG(Ar, d2, d9)
    final_vertices = funcs.calculate_sh_vertices(coeffs, base_mesh_data["sph_coords"])
    theta = np.deg2rad(incident_angle)
    rot_y = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    final_vertices = final_vertices @ rot_y.T

    return {"vertices": final_vertices, "faces": base_mesh_data["faces"]}


if __name__ == "__main__":
    import os
    from itertools import product

    params = {"Ar": [0.5, 2.5], "d2": 0.0, "d9": 0.0, "incident_angle": [0, 45, 90]}

    DATASET_DIR = "dataset_test"
    GEOMETRY_DIR = os.path.join(DATASET_DIR, "stl")
    os.makedirs(GEOMETRY_DIR, exist_ok=True)

    bash_mesh = create_base_sh_mesh(level=4)

    for Ar, ang in product(params["Ar"], params["incident_angle"]):
        geom = generate_sh_particle(
            Ar=Ar,
            d2=params["d2"],
            d9=params["d9"],
            base_mesh_data=bash_mesh,
            incident_angle=ang,
        )

        filename = f"particle_Ar{Ar:.2f}_ang{ang:.1f}.stl"
        file_path = os.path.join(GEOMETRY_DIR, filename)

        funcs.save_stl_from_data(file_path, geom["vertices"], geom["faces"])
        print("Saved:", file_path)
