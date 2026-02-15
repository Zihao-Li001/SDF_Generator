# root/data/config.py
# config for Generator


class DatasetConfig:
    # ===== Geometry Parameter =====
    GEOM_PARAM_RANGES = {
        "aspect_ratio": (0.5, 2.5),  # aspect_ratio
        "d2": (0.0, 0.0001),
        "d9": (0.0, 0.0001),
    }

    # ===== Flow Parameter =====
    FLOW_PARAM_RANGES = {
        "incident_angle": (0, 90),  # incident angle
        "reynolds_number": (10, 200),  # Reynold number
    }

    # ===== Sampling =====
    SAMPLING = {
        "n_geometries": 200,  # total number of geometries
        "n_flow_per_geometry": 20,  # number of flow condition for each geom
        "mode": "lhs",  # Sampling Mode: lhs/random/physics
        "lhs_criterion": "maximin",  # LHS Optimization Criterion
    }

    # ===== Computation =====
    COMPUTATION = {
        "mesh_level": 4,  # Fine Level of stl
        "voxel_resolution": 64,  # Resolution of voxel
        "sdf_resolution": 64,  # Resolution of sdf
    }

    # ===== Output =====
    OUTPUT = {
        "dataset_dir": "dataset",
        "stl_dir": "stl",
        "voxel_dir": "voxel",
        "sdf_dir": "sdf",
        "metadata_dir": "metadata.csv",
    }

    @property
    def total_samples(self):
        return self.SAMPLING["n_geometries"] * self.SAMPLING["n_flow_per_geometry"]

    def get_param_range(self):
        return {**self.GEOM_PARAM_RANGES, **self.FLOW_PARAM_RANGES}


CONFIG = DatasetConfig()
