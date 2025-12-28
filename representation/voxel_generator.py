from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import trimesh


@dataclass
class VoxelConfig:
    resolution: int = 64
    binary_threshold: float = 0.5
    # Reference extent is the size of entire box
    reference_extent: float = 2.0


class VoxelProcessor:
    def __init__(self, config: VoxelConfig):
        self.config = config
        self.stats: Dict[str, Any] = {}

    def ensure_binary_voxel(self, voxel: np.ndarray) -> np.ndarray:
        """Convert voxel array to strict binary values."""
        return (voxel > self.config.binary_threshold).astype(np.float32)

    def center_voxel(self, voxel: np.ndarray, target_size: int) -> np.ndarray:
        # find the bounds of non-zore voxel
        non_zero_indices = np.argwhere(voxel > 0)
        if len(non_zero_indices) == 0:
            return np.zeros((target_size, target_size, target_size), dtype=np.float32)

        min_coords = non_zero_indices.min(axis=0)
        max_coords = non_zero_indices.max(axis=0)
        shape = max_coords - min_coords + 1

        # check over bounds
        source_slice = tuple(slice(min_coords[i], max_coords[i] + 1) for i in range(3))
        cropped_voxel = voxel[source_slice]

        centered_voxel = np.zeros(
            (target_size, target_size, target_size), dtype=np.float32
        )
        target_min = (np.array([target_size] * 3) - shape) // 2
        target_slice = tuple(
            slice(target_min[i], target_min[i] + shape[i]) for i in range(3)
        )
        centered_voxel[target_slice] = cropped_voxel

        return self.ensure_binary_voxel(centered_voxel)

    def voxelize_particle(self, mesh: trimesh.Trimesh) -> np.ndarray:
        if mesh.is_empty or len(mesh.vertices) == 0:
            raise ValueError("Empty or Invalid mesh")

        # copy the mesh, then translate the copied mesh to the centroid
        mesh_copy = mesh.copy()
        mesh_copy.apply_translation(-mesh.centroid)

        pitch = self.config.reference_extent / self.config.resolution

        voxel_grid = mesh_copy.voxelized(pitch=pitch, method="subdivide")
        voxel_grid = voxel_grid.fill()
        voxel = voxel_grid.matrix.astype(np.float32)

        return self.center_voxel(voxel, self.config.resolution)

    def save_npy(self, voxel: np.ndarray, path: Path) -> None:
        if voxel.shape != (self.config.resolution,) * 3:
            raise ValueError(f"Invalid voxel shape: {voxel.shape}")
        np.save(path, voxel.astype(np.float32))

    def process_stl_file(self, stl_path: Path, output_dir: Path) -> List[Path]:
        """Process single STL file with rotations."""
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()

        voxel = self.voxelize_particle(mesh)
        output_files = []

        # Save original
        out_path = output_dir / f"{stl_path.stem}.npy"
        self.save_npy(voxel, out_path)
        output_files.append(out_path)
        return output_files


def process_voxel(stl_filepath: str, output_filepath: str, resolution: int = 64):
    """
    Args:
        stl_filepath (str): 输入的STL文件路径。
        output_filepath (str): 输出的Voxel文件路径 (应以 .npy 结尾)。
        resolution (int): Voxel的分辨率。
    """
    try:
        stl_path = Path(stl_filepath)
        output_path = Path(output_filepath)

        config = VoxelConfig(resolution=resolution)
        processor = VoxelProcessor(config)

        mesh_or_scene = trimesh.load(stl_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            mesh = mesh_or_scene.dump().sum()
        else:
            mesh = mesh_or_scene

        if not hasattr(mesh, "is_empty") or mesh.is_empty:
            print(f"警告: 来自 {stl_filepath} 的网格为空或无效，已跳过。")
            return

        voxel_data = processor.voxelize_particle(mesh)

        processor.save_npy(voxel_data, output_path)

    except Exception as e:
        import traceback

        print(f" ConvertError: {stl_filepath}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("--- Runing test ---")

    test_input = Path("./test_stl")
    test_output = Path("./test_output")
    test_stl_path = test_input / "test_sphere.stl"
