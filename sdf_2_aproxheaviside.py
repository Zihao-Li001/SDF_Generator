import numpy as np
import os, io, zlib, glob
from mysdftools.io import load_compressed_npy

# Set input and output directories
INPUT_DIR = "./dataset/sdf/"
OUTPUT_DIR = "./dataset/heaviside/"
EPS = 0.02
COMPRESSION_LEVEL = 6
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sdf_to_heaviside_poly(sdf: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    H(x) = 1/2( (f+D)/D + (1/pi)*sim(pi*f/D))
    Args:
        sdf: numpy ndarray, signed distance field
        eps: smoothing width, larger eps for smoother result
    """
    H = np.zeros_like(sdf, dtype=np.float32)

    H[sdf > eps] = 1.0
    mask_mid = np.abs(sdf) <= eps
    x = sdf[mask_mid]

    H[mask_mid] = 0.5 * ((x + eps) / eps + (1 / np.pi) * np.sin(np.pi * x / eps))

    return H


def load_file(filepath):
    return load_compressed_npy(filepath).astype(np.float32)


def process_one_sdf(npy_z_path):
    base = os.path.basename(npy_z_path)
    name = os.path.splitext(base)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}.z")

    if os.path.exists(out_path):
        return

    sdf = load_compressed_npy(npy_z_path).astype(np.float32)

    H = sdf_to_heaviside_poly(sdf, eps=EPS)

    buffer = io.BytesIO()
    np.save(buffer, H)
    buffer.seek(0)

    compressed = zlib.compress(buffer.read(), level=COMPRESSION_LEVEL)

    with open(out_path, "wb") as f:
        f.write(compressed)

    print("Saved ->", out_path)


def main():
    sdf_files = glob.glob(os.path.join(INPUT_DIR, "*.npy.z"))
    print(f"Found {len(sdf_files)} sdf files.")
    for path in sdf_files:
        process_one_sdf(path)


if __name__ == "__main__":
    main()
