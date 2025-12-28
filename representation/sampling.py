from config import CONFIG

import numpy as np
from pyDOE2 import lhs
from typing import Tuple, List, Dict, Optional
import sys
import os

# IMPORTANT: ensure relative import works in your package layout
from .nonuniform_flow_sampler import NonUniformFlowSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParameterSampler:
    """
    Final, unambiguous sampler.

    Sampling modes (config.SAMPLING["mode"]):
      - "random": geometry random, flow random
      - "lhs":    geometry LHS,    flow LHS
      - "physics": geometry random (default), flow NonUniformFlowSampler (physics-informed)

    Notes:
      - NonUniformFlowSampler generates directly scaled values (angle/Re in physical units).
      - LHS/random flows are generated in unit hypercube then scaled via _scale_samples.
    """

    def __init__(self, config):
        self.config = config

        self.geom_param_ranges = config.GEOM_PARAM_RANGES
        self.flow_param_ranges = config.FLOW_PARAM_RANGES
        self.geom_param_names = list(self.geom_param_ranges.keys())
        self.flow_param_names = list(self.flow_param_ranges.keys())

        self.mode = self.config.SAMPLING["mode"]

        # physics flow sampler only when needed
        self.flow_sampler: Optional[NonUniformFlowSampler] = None
        if self.mode == "physics":
            self.flow_sampler = self._build_physics_flow_sampler()

    # -------------------------
    # helpers
    # -------------------------
    def _build_physics_flow_sampler(self) -> NonUniformFlowSampler:
        # Expected keys in FLOW_PARAM_RANGES
        if "incident_angle" not in self.flow_param_ranges:
            raise KeyError(
                "FLOW_PARAM_RANGES must contain 'incident_angle' for physics mode"
            )
        if "reynolds_number" not in self.flow_param_ranges:
            raise KeyError(
                "FLOW_PARAM_RANGES must contain 'reynolds_number' for physics mode"
            )

        return NonUniformFlowSampler(
            n_samples=self.config.SAMPLING["n_flow_per_geometry"],
            angle_range=self.flow_param_ranges["incident_angle"],
            re_range=self.flow_param_ranges["reynolds_number"],
        )

    def _scale_samples(
        self,
        samples: np.ndarray,
        param_ranges: Dict[str, Tuple[float, float]],
        param_names: List[str],
    ) -> np.ndarray:
        """
        Scale samples from unit hypercube [0,1] to physical ranges.
        """
        scaled = np.empty_like(samples, dtype=float)
        for i, param in enumerate(param_names):
            low, high = param_ranges[param]
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def _generate_lhs_samples(self, n_params: int, n_samples: int) -> np.ndarray:
        return lhs(
            n_params,
            samples=n_samples,
            criterion=self.config.SAMPLING.get("lhs_criterion", "maximin"),
        )

    def _generate_random_samples(self, n_params: int, n_samples: int) -> np.ndarray:
        return np.random.random((n_samples, n_params))

    # -------------------------
    # geometry sampling
    # -------------------------
    def generate_geometry_samples(self) -> np.ndarray:
        n_geom = self.config.SAMPLING["n_geometries"]
        n_geom_params = len(self.geom_param_ranges)

        if self.mode == "lhs":
            geom_unit = self._generate_lhs_samples(n_geom_params, n_geom)
        else:
            # "random" and "physics" default to random geometry sampling
            # if you want "physics" geometry to be lhs, change this branch accordingly
            geom_unit = self._generate_random_samples(n_geom_params, n_geom)

        return self._scale_samples(
            geom_unit, self.geom_param_ranges, self.geom_param_names
        )

    # -------------------------
    # flow sampling
    # -------------------------
    def _generate_flow_samples_lhs_or_random(self) -> np.ndarray:
        n_flow = self.config.SAMPLING["n_flow_per_geometry"]
        n_flow_params = len(self.flow_param_ranges)

        if self.mode == "lhs":
            flow_unit = self._generate_lhs_samples(n_flow_params, n_flow)
        else:
            flow_unit = self._generate_random_samples(n_flow_params, n_flow)

        return self._scale_samples(
            flow_unit, self.flow_param_ranges, self.flow_param_names
        )

    def _generate_flow_samples_physics(self, geom_idx: int) -> np.ndarray:
        assert (
            self.flow_sampler is not None
        ), "flow_sampler must be initialized in physics mode"

        flow_2d = self.flow_sampler.sample(geom_idx)  # shape: (n, 2) -> [angle, re]
        out = np.zeros((flow_2d.shape[0], len(self.flow_param_names)), dtype=float)

        # Map to configured flow param order
        for i, name in enumerate(self.flow_param_names):
            if name == "incident_angle":
                out[:, i] = flow_2d[:, 0]
            elif name == "reynolds_number":
                out[:, i] = flow_2d[:, 1]
            else:
                raise KeyError(
                    f"physics mode supports only 'incident_angle' and 'reynolds_number'. "
                    f"Unsupported: {name}"
                )
        return out

    def generate_single_flow_sample_set(self, geom_idx: int) -> np.ndarray:
        if self.mode == "physics":
            return self._generate_flow_samples_physics(geom_idx)
        return self._generate_flow_samples_lhs_or_random()

    def generate_all_flow_samples(self, n_geometries: int) -> List[np.ndarray]:
        return [self.generate_single_flow_sample_set(i) for i in range(n_geometries)]

    # -------------------------
    # public APIs
    # -------------------------
    def sample(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        geometry_samples = self.generate_geometry_samples()
        n_geometries = len(geometry_samples)
        flow_samples_list = self.generate_all_flow_samples(n_geometries)
        return geometry_samples, flow_samples_list

    def generate_sample(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.sample()

    def get_sample_info(self):
        return {
            "n_geometries": self.config.SAMPLING["n_geometries"],
            "n_flow_per_geometry": self.config.SAMPLING["n_flow_per_geometry"],
            "total_samples": getattr(self.config, "total_samples", None),
            "sampling_mode": self.mode,
            "geometry_parameters": self.geom_param_names,
            "flow_parameters": self.flow_param_names,
            "geometry_ranges": self.geom_param_ranges,
            "flow_ranges": self.flow_param_ranges,
        }

    def validate_config(self) -> bool:
        try:
            required_keys = ["n_geometries", "n_flow_per_geometry", "mode"]
            for key in required_keys:
                if key not in self.config.SAMPLING:
                    raise KeyError(f"Missing required key in SAMPLING: {key}")

            if self.config.SAMPLING["n_geometries"] <= 0:
                raise ValueError("n_geometries must be positive")
            if self.config.SAMPLING["n_flow_per_geometry"] <= 0:
                raise ValueError("n_flow_per_geometry must be positive")

            if self.config.SAMPLING["mode"] not in ["lhs", "random", "physics"]:
                raise ValueError(
                    "Sampling mode must be one of: 'lhs', 'random', 'physics'"
                )

            # validate ranges
            for param, (low, high) in self.geom_param_ranges.items():
                if low >= high:
                    raise ValueError(
                        f"Invalid geometry range for {param}: {low} >= {high}"
                    )

            for param, (low, high) in self.flow_param_ranges.items():
                if low >= high:
                    raise ValueError(f"Invalid flow range for {param}: {low} >= {high}")

            # physics mode must have expected keys
            if self.config.SAMPLING["mode"] == "physics":
                if "incident_angle" not in self.flow_param_ranges:
                    raise KeyError(
                        "physics mode requires FLOW_PARAM_RANGES['incident_angle']"
                    )
                if "reynolds_number" not in self.flow_param_ranges:
                    raise KeyError(
                        "physics mode requires FLOW_PARAM_RANGES['reynolds_number']"
                    )

                # physics mode currently expects only these two flow params
                allowed = {"incident_angle", "reynolds_number"}
                extra = set(self.flow_param_names) - allowed
                if extra:
                    raise ValueError(
                        f"physics mode supports only {sorted(allowed)}. "
                        f"Found extra flow params: {sorted(extra)}"
                    )

            return True

        except (KeyError, ValueError) as e:
            print(f"Configuration validation error: {e}")
            return False


if __name__ == "__main__":
    config = CONFIG
    sampler = ParameterSampler(config)

    if sampler.validate_config():
        print("Configuration is valid!")
        geom_samples, flow_samples_list = sampler.sample()

        print(f"Generated {len(geom_samples)} geometry samples")
        print(f"Generated flow samples for {len(flow_samples_list)} geometries")
        print(f"Each geometry has {len(flow_samples_list[0])} flow conditions")

        info = sampler.get_sample_info()
        print("\nSample Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("Configuration validation failed!")
