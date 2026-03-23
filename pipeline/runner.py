from __future__ import annotations

import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import CONFIG
from derived_fields import (
    build_sample_context,
    compute_derived_fields,
    get_default_derived_field_providers,
    get_derived_fieldnames,
)
from pipeline.sample_record import BASE_METADATA_FIELDS, build_metadata_record
from . import worker_state
from representation.funcs import add_gaussian_noise
from representation.funcs import rotate_vertices as rotate_vertices
from representation.funcs import save_stl_from_data
from representation.geom_generator import create_base_sh_mesh, generate_sh_particle
from representation.sdf_generator import process_sdf
from representation.sampling import ParameterSampler
from representation.voxel_generator import process_voxel

DATASET_ROOT = Path(CONFIG.OUTPUT["dataset_dir"]).resolve()
STL_DIR = DATASET_ROOT / CONFIG.OUTPUT["stl_dir"]
VOXEL_DIR = DATASET_ROOT / CONFIG.OUTPUT["voxel_dir"]
SDF_DIR = DATASET_ROOT / CONFIG.OUTPUT["sdf_dir"]
METADATA_PATH = DATASET_ROOT / CONFIG.OUTPUT["metadata_dir"]


def ensure_output_dirs(enable_voxel: bool, enable_sdf: bool) -> None:
    STL_DIR.mkdir(parents=True, exist_ok=True)
    if enable_voxel:
        VOXEL_DIR.mkdir(parents=True, exist_ok=True)
    if enable_sdf:
        SDF_DIR.mkdir(parents=True, exist_ok=True)


def metadata_fieldnames(derived_field_providers) -> List[str]:
    return [
        *BASE_METADATA_FIELDS[:6],
        *get_derived_fieldnames(derived_field_providers),
        *BASE_METADATA_FIELDS[6:],
    ]


def build_artifact_paths(dataset_root: Path, geom_id: int, rotate_id: int) -> Tuple[Path, Path, Path]:
    sample_id = geom_id * 1000 + rotate_id
    stl_dir = dataset_root / worker_state.CONFIG.OUTPUT["stl_dir"]
    voxel_dir = dataset_root / worker_state.CONFIG.OUTPUT["voxel_dir"]
    sdf_dir = dataset_root / worker_state.CONFIG.OUTPUT["sdf_dir"]
    return (
        stl_dir / f"{sample_id}.stl",
        voxel_dir / f"{sample_id}.npy",
        sdf_dir / f"{sample_id}.npy.z",
    )


def generate_records_for_geometry(idx_and_params):
    idx, params = idx_and_params
    ar, d2, d9 = params

    geometry = generate_sh_particle(ar, d2, d9, worker_state.BASE_MESH)
    if worker_state.ADD_NOISE:
        geometry["vertices"] = add_gaussian_noise(geometry["vertices"], scale=0.01)

    dataset_root = Path(worker_state.CONFIG.OUTPUT["dataset_dir"]).resolve()
    records = []

    for fidx, flow_params in enumerate(worker_state.FLOW_PARAMS_LIST[idx]):
        geom_id = idx + 1
        rotate_id = fidx + 1
        angle, re = flow_params

        rotated_vertices = rotate_vertices(geometry["vertices"], 90 - angle, axis="y")
        stl_path, voxel_path, sdf_path = build_artifact_paths(
            dataset_root, geom_id, rotate_id
        )
        context = build_sample_context(
            dataset_root=dataset_root,
            geom_id=geom_id,
            rotate_id=rotate_id,
            aspect_ratio=ar,
            d2=d2,
            d9=d9,
            incident_angle=angle,
            reynolds_number=re,
            stl_path=stl_path,
            voxel_path=voxel_path,
            sdf_path=sdf_path,
        )

        save_stl_from_data(str(context.stl_path), rotated_vertices, geometry["faces"])

        if worker_state.ENABLE_VOXEL:
            process_voxel(
                str(context.stl_path),
                str(context.voxel_path),
                worker_state.CONFIG.COMPUTATION["voxel_resolution"],
            )
        if worker_state.ENABLE_SDF:
            process_sdf(
                str(context.stl_path),
                str(context.sdf_path),
                worker_state.CONFIG.COMPUTATION["sdf_resolution"],
            )

        derived_outputs = compute_derived_fields(
            context,
            worker_state.DERIVED_FIELD_PROVIDERS,
        )
        records.append(build_metadata_record(context, derived_outputs))

    return records


def run_dataset_generation(
    enable_voxel: bool = False,
    enable_sdf: bool = False,
    add_noise_to_geom: bool = False,
) -> None:
    print("Start to generate dataset...")
    ensure_output_dirs(enable_voxel, enable_sdf)

    print("Sampling...")
    sampler = ParameterSampler(CONFIG)
    if not sampler.validate_config():
        print("=== Error! Configuration validation failed! Exiting... ===")
        return

    sample_info = sampler.get_sample_info()
    print(f"    Generating {sample_info['n_geometries']} Geometries")
    print(f"    Total samples: {sample_info['total_samples']}")
    print("Sampling Finish")

    geom_params, flow_params_list = sampler.generate_sample()
    base_mesh = create_base_sh_mesh(level=CONFIG.COMPUTATION["mesh_level"])
    derived_field_providers = get_default_derived_field_providers()

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=metadata_fieldnames(derived_field_providers),
        )
        writer.writeheader()

        tasks = [(i, geom_params[i]) for i in range(len(geom_params))]
        max_workers = CONFIG.COMPUTATION.get("num_workers", cpu_count() - 1)
        num_workers = max(1, min(cpu_count(), max_workers))

        with Pool(
            processes=num_workers,
            initializer=worker_state.init_worker,
            initargs=(
                base_mesh,
                flow_params_list,
                CONFIG,
                enable_voxel,
                enable_sdf,
                add_noise_to_geom,
                derived_field_providers,
            ),
        ) as pool:
            with tqdm(total=len(geom_params), desc="Generating Geometries") as progress_bar:
                for records in pool.imap_unordered(generate_records_for_geometry, tasks):
                    for record in records:
                        writer.writerow(record)
                    progress_bar.update(1)

    print(f"METADATA saved to {METADATA_PATH}")
    print("Finished generating dataset.")
