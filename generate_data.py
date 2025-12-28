# root/data/generator.py

import os
import argparse
from tqdm import tqdm
import csv
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

# geometry generator tool
from drag_coeff import calculate_drag_coefficient
from config import CONFIG
from representation.sampling import ParameterSampler
from representation.geom_generator import generate_sh_particle, create_base_sh_mesh
from representation.funcs import rotate_vertices as R
from representation.funcs import add_gaussian_noise
from representation.funcs import save_stl_from_data
from representation.voxel_generator import process_voxel
from representation.sdf_generator import process_sdf
from representation.calc_geom_metadata import compute_geom_info, FLOW_DIR

DATASET_ROOT = Path(CONFIG.OUTPUT["dataset_dir"]).resolve()
STL_DIR = DATASET_ROOT / CONFIG.OUTPUT["stl_dir"]
VOXEL_DIR = DATASET_ROOT / CONFIG.OUTPUT["voxel_dir"]
SDF_DIR = DATASET_ROOT / CONFIG.OUTPUT["sdf_dir"]
METADATA_PATH = DATASET_ROOT / CONFIG.OUTPUT["metadata_dir"]


class SuppressPrints:
    # Suppress the Prints from VAE generator
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def generate_one(args):
    (
        idx,
        params,
        base_mesh,
        config,
        enable_voxel,
        enable_sdf,
        add_noise,
        flow_params_list,
    ) = args
    ar, d2, d9 = params
    geom = generate_sh_particle(ar, d2, d9, base_mesh)
    if add_noise:
        # add noise to the geomtry
        geom["vertices"] = add_gaussian_noise(geom["vertices"], scale=0.01)

    records = []
    dataset_root = Path(config.OUTPUT["dataset_dir"]).resolve()
    stl_dir = dataset_root / config.OUTPUT["stl_dir"]
    voxel_dir = dataset_root / config.OUTPUT["voxel_dir"]
    sdf_dir = dataset_root / config.OUTPUT["sdf_dir"]

    for fidx, flow_params in enumerate(flow_params_list[idx]):
        geom_id = idx + 1
        rotate_id = fidx + 1
        sample_id = geom_id * 1000 + rotate_id

        angle, re = flow_params
        rotated_vertices = R(geom["vertices"], 90 - angle, axis="y")
        Cd_eq = calculate_drag_coefficient(re, ar, angle)

        stl_path = stl_dir / f"{sample_id}.stl"
        voxel_path = voxel_dir / f"{sample_id}.npy"
        sdf_path = sdf_dir / f"{sample_id}.npy"

        save_stl_from_data(str(stl_path), rotated_vertices, geom["faces"])
        try:
            _, d_eq, a_ref = compute_geom_info(Path(stl_path), FLOW_DIR)
        except Exception as e:
            print(f"[Warn] cannot compute geom info. Error: {e}")
            d_eq, a_ref = float("nan"), float("nan")
        if enable_voxel:
            process_voxel(stl_path, voxel_path, CONFIG.COMPUTATION["voxel_resolution"])
        if enable_sdf:
            process_sdf(stl_path, sdf_path, CONFIG.COMPUTATION["sdf_resolution"])

        records.append(
            {
                "sample_id": sample_id,
                "geom_id": geom_id,
                "rotate_id": rotate_id,
                "aspect_ratio": ar,
                "incident_angle": angle,
                "lRef": d_eq,
                "Aref": a_ref,
                "Re": re,
                "Cd_equation": Cd_eq,
                "stl_path": stl_path.relative_to(dataset_root).as_posix(),
                "voxel_path": voxel_path.relative_to(dataset_root).as_posix(),
                "sdf_path": sdf_path.relative_to(dataset_root).as_posix(),
                # "stl_path": stl_path,
                # "voxel_path": voxel_path,
                # "sdf_path": sdf_path,
            }
        )
    return records


def main(enable_voxel=False, enable_sdf=False, add_noise_to_geom=False):
    print("Start to generate dataset...")

    os.makedirs(STL_DIR, exist_ok=True)
    if enable_voxel:
        os.makedirs(VOXEL_DIR, exist_ok=True)
    if enable_sdf:
        os.makedirs(SDF_DIR, exist_ok=True)

    print("Sampling...")
    sampler = ParameterSampler(CONFIG)
    if not sampler.validate_config():
        print("=== Error! Configuration validation failed! Exiting... ===")
        return
    else:
        sample_info = sampler.get_sample_info()
        print(f"    Generating {sample_info['n_geometries']} Geometries")
        print(f"    Total samples: {sample_info['total_samples']}")
        print("Sampling Finsih")

    geom_params, flow_params_list = sampler.generate_sample()
    base_mesh = create_base_sh_mesh(level=CONFIG.COMPUTATION["mesh_level"])

    fieldnames = [
        "sample_id",
        "geom_id",
        "rotate_id",
        "aspect_ratio",
        "incident_angle",
        "lRef",
        "Aref",
        "Re",
        "Cd_equation",
        "stl_path",
        "voxel_path",
        "sdf_path",
    ]

    # generate geometry while write the metadate.csv
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=len(geom_params), desc="Generating Geometries")
        tasks = [
            (
                i,
                geom_params[i],
                base_mesh,
                CONFIG,
                enable_voxel,
                enable_sdf,
                add_noise_to_geom,
                flow_params_list,
            )
            for i in range(len(geom_params))
        ]

        num_workers = max(1, cpu_count() // 6)
        with Pool(processes=num_workers) as pool:
            for records in pool.imap_unordered(generate_one, tasks):
                for record in records:
                    writer.writerow(record)
                pbar.update(1)
                pbar.refresh()
        pbar.close()

    print(f"METADATA saved to {METADATA_PATH}")
    print("Finished generating dataset.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-voxel", action="store_true")
    parser.add_argument("--enable-sdf", action="store_true")
    parser.add_argument("--noise", action="store_true", dest="add_noise_to_geom")
    return parser.parse_args()


if __name__ == "__main__":
    print("=============================================")
    print("      Dataset Generating           ")
    print("=============================================")

    try:
        args = parse_args()
        main(
            args.enable_voxel,
            args.enable_sdf,
            args.add_noise_to_geom,
        )

    except Exception as e:
        print(f" Error occured: {e}")
        import traceback

        traceback.print_exc()
