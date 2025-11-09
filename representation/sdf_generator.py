# VAE-ANN/core/representation/sdf_generator.py
import numpy as np
import trimesh
import os


class SDFGenerator:
    """
    Generate a Signed Distances Function (SDF) grid from a 3D geometry
    Two modes:
    1. Dynamic bounds: Create a grid that fits particle with padding.
    2. Fixed bounds: Create a grid of a specificed size.
    """

    def __init__(
        self, resolution: int = 32, padding: float = 0.1, box_size: float = 2.0
    ):
        """
        Initialization
        Args:
            resolution(int)
            padding(float): Padding factor for dynamic bounds
            box_size(float): Fixed box size for fixed bounds mode
        """
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")
        self.resolution = resolution
        self.padding = padding
        self.box_size = box_size
        self.mesh = None

    def load_mesh(self, mesh_file_path: str) -> trimesh.Trimesh:
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

        # print(f"Loading mesh file: {mesh_file_path}...")
        try:
            mesh = trimesh.load_mesh(mesh_file_path)
            if not mesh.is_watertight:
                print("Warning: Mesh is not watertight. Attempting to repair.")
                mesh.fill_holes()
                if not mesh.is_watertight:
                    raise ValueError("The loaded STL mesh is not watertight.")
            self.mesh = mesh
            return self.mesh
        except Exception as e:
            raise Exception(f"Failed to load STL file {mesh_file_path}: {e}")

    def _center_mesh(
        self, geometry_mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        centered_mesh = geometry_mesh.copy()
        original_center = centered_mesh.bounds.mean(axis=0)
        centered_mesh.apply_translation(-original_center)
        final_center = centered_mesh.bounds.mean(axis=0)

        assert np.allclose(
            final_center, [0, 0, 0], atol=1e-5
        ), f"Centering failed, Final center: {final_center}"

        return centered_mesh, original_center

    def generate_sdf(self, input_mesh: trimesh.Trimesh) -> dict:
        # print("Centering mesh...")
        centered_mesh, original_center = self._center_mesh(input_mesh)

        if self.box_size:
            # print(f"Using fixed box size: {self.box_size}")
            half_box = self.box_size / 2.0
            min_bounds = np.array([-half_box, -half_box, -half_box])
            max_bounds = np.array([half_box, half_box, half_box])

            mesh_extent = np.max(centered_mesh.bounds[1] - centered_mesh.bounds[0])
            if mesh_extent > self.box_size:
                print(
                    f"Warning: Mesh extent ({mesh_extent:.3f}) exceeds box size {self.box_size}"
                )
        else:
            mesh_bounds = centered_mesh.bounds
            dims = mesh_bounds[1] - mesh_bounds[0]
            min_bounds = mesh_bounds[0] - dims * self.padding
            max_bounds = mesh_bounds[1] + dims * self.padding

        x_coords = np.linspace(min_bounds[0], max_bounds[0], self.resolution)
        y_coords = np.linspace(min_bounds[1], max_bounds[1], self.resolution)
        z_coords = np.linspace(min_bounds[2], max_bounds[2], self.resolution)

        grid_x, grid_y, grid_z = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )
        query_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        # print(f"Generating SDF for {len(query_points)} points",
        # f"at {self.resolution}^3 resolution")
        distances = trimesh.proximity.signed_distance(centered_mesh, query_points)
        distances = -distances
        sdf_grid = distances.reshape(self.resolution, self.resolution, self.resolution)
        # print("SDF genertation complete.")

        return {
            "sdf_grid": sdf_grid,
            "original_center": original_center,
            "grid_bounds": np.array([min_bounds, max_bounds]),
        }

    @staticmethod
    def save_sdf_to_npy(sdf_data: np.ndarray, output_filepath: str):
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(output_filepath, sdf_data)
        # print(f"SDF data saved to {output_filepath}")


def process_sdf(
    input_filepath: str,
    output_filepath: str,
    resolution: int,
    padding: float = 0.0,
    box_size: float = 2.0,
):
    """
    Main processing pipeline to generate and save an SDF from a mesh file.
    """
    try:
        sdf_gen = SDFGenerator(
            resolution=resolution, padding=padding, box_size=box_size
        )

        mesh = sdf_gen.load_mesh(input_filepath)
        sdf_results = sdf_gen.generate_sdf(mesh)
        sdf_gen.save_sdf_to_npy(sdf_results["sdf_grid"], output_filepath)

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"An error during SDF processing {input_filepath}: {e}")
