import numpy as np
import trimesh
import skimage.measure
import os

# --- Configuration ---
# Set the path to the .npy.z file you want to visualize
SDF_FILE_PATH = "./dataset/voxel/particle_0000_02.npy"

# The surface of a Signed Distance Field is at level 0
CONTOUR_LEVEL = 0.0
# --- End Configuration ---


# --- Main Visualization Logic ---
if __name__ == "__main__":
    if not os.path.exists(SDF_FILE_PATH):
        print(f"Error: File not found at '{SDF_FILE_PATH}'")
    else:
        print(f"Loading and decompressing {SDF_FILE_PATH}...")
        # 1. Load the compressed SDF voxel grid
        voxels = np.load(SDF_FILE_PATH)

        print(f"Loaded voxel grid with shape: {voxels.shape}")
        print("Running Marching Cubes to extract surface...")

        # 2. Extract surface mesh using Marching Cubes
        # (This is the same as your example code)
        vertices, faces, normals, _ = skimage.measure.marching_cubes(
            voxels, level=CONTOUR_LEVEL
        )

        print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces.")

        # 3. Create a trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        # 4. Show the mesh in an interactive window
        print("Displaying mesh...")
        mesh.show()
