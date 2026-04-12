from __future__ import annotations

BASE_MESH = None
FLOW_PARAMS_LIST = None
CONFIG = None
ENABLE_VOXEL = None
ENABLE_SDF = None
ADD_NOISE = None
DERIVED_FIELD_PROVIDERS = None
GEOM_ID_START = None


def init_worker(
    base_mesh,
    flow_params_list,
    config,
    enable_voxel,
    enable_sdf,
    add_noise,
    derived_field_providers,
    geom_id_start,
):
    global BASE_MESH, FLOW_PARAMS_LIST, CONFIG
    global ENABLE_VOXEL, ENABLE_SDF, ADD_NOISE, DERIVED_FIELD_PROVIDERS
    global GEOM_ID_START

    BASE_MESH = base_mesh
    FLOW_PARAMS_LIST = flow_params_list
    CONFIG = config
    ENABLE_VOXEL = enable_voxel
    ENABLE_SDF = enable_sdf
    ADD_NOISE = add_noise
    DERIVED_FIELD_PROVIDERS = derived_field_providers
    GEOM_ID_START = geom_id_start
