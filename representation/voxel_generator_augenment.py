import numpy as np
import trimesh
import zlib
import tarfile
from pathlib import Path
from scipy.ndimage import rotate


class Config:
    resolution = (32, 32, 32)
    scale_factor = 1.0
    rotation_angles = [-45]
    scale_factors = [0.8, 1.0, 1.2]


class VoxelProcessor:
    def __init__(self, config: Config):
        self.config = config

    def voxelize_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounding_box.center_mass)
        max_extent = np.max(mesh.extents)
        scale = self.config.scale_factor / max_extent
        mesh.apply_scale(scale)

        voxelized = mesh.voxelized(pitch=1.0 / self.config.resolution[0])
        filled = voxelized.fill()
        matrix = filled.matrix.astype(np.uint8)

        return self.crop_or_pad_voxel(matrix, self.config.resolution)

    def crop_or_pad_voxel(self, voxel: np.ndarray, target_shape: tuple) -> np.ndarray:
        current_shape = voxel.shape
        crop = [slice(0, min(c, t)) for c, t in zip(current_shape, target_shape)]
        voxel = voxel[tuple(crop)]

        padded = np.zeros(target_shape, dtype=np.uint8)
        slices = tuple(slice(0, s) for s in voxel.shape)
        padded[slices] = voxel

        return padded

    def rotate_voxel(self, voxel: np.ndarray, angle: float) -> np.ndarray:
        rotated = rotate(
            voxel, angle, axes=(1, 2), reshape=False, order=0, mode="constant", cval=0
        )
        return (rotated > 0.5).astype(np.uint8)

    def save_compressed(self, array: np.ndarray, output_path: Path):
        compressed = zlib.compress(array.tobytes())
        with open(output_path, "wb") as f:
            f.write(compressed)

    def process_stl_file(self, stl_path: Path, output_dir: Path):
        mesh = trimesh.load_mesh(stl_path)
        for scale in self.config.scale_factors:
            mesh_copy = mesh.copy()
            max_extent = np.max(mesh_copy.extents)
            mesh_copy.apply_translation(-mesh_copy.bounding_box.center_mass)
            mesh_copy.apply_scale(scale / max_extent)

            base_voxel = self.voxelize_mesh(mesh_copy)

            for angle in self.config.rotation_angles:
                voxel = self.rotate_voxel(base_voxel, angle)
                voxel = self.crop_or_pad_voxel(voxel, self.config.resolution)

                filename = f"{stl_path.stem}_s{scale:.1f}_r{angle}.npy.z"
                output_path = output_dir / filename
                self.save_compressed(voxel, output_path)


def archive_output(output_dir: Path, tar_path: Path):
    with tarfile.open(tar_path, "w") as tar:
        for file_path in output_dir.glob("*.npy.z"):
            tar.add(file_path, arcname=file_path.name)


def process_directory(input_dir: Path, output_dir: Path, tar_output_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = VoxelProcessor(Config())

    for file in input_dir.glob("*.stl"):
        processor.process_stl_file(file, output_dir)

    archive_output(output_dir, tar_output_path)


if __name__ == "__main__":
    input_stl_dir = Path("input_stl")
    output_voxel_dir = Path("output_voxel")
    tar_output = Path("output_voxel/voxel_data.tar")

    process_directory(input_stl_dir, output_voxel_dir, tar_output)
