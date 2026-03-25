from config import CONFIG

import numpy as np
from pyDOE2 import lhs
from typing import Callable, Dict, List, Tuple

UnitSampler = Callable[[int, int], np.ndarray]


class ParameterSampler:
    """
    Sampler orchestration layer with pluggable mode strategies.

    Supported modes:
      - "random": geometry random, flow random
      - "lhs":    geometry LHS, flow LHS

    Extensibility:
      - Add new modes via register_mode().
      - Each mode provides one unit-hypercube sampler for geometry and one for flow.
      - Scaling from [0, 1] to physical ranges is handled centrally by _scale_samples().
    """

    def __init__(self, config):
        self.config = config

        self.geom_param_ranges = config.GEOM_PARAM_RANGES
        self.flow_param_ranges = config.FLOW_PARAM_RANGES
        self.geom_param_names = list(self.geom_param_ranges.keys())
        self.flow_param_names = list(self.flow_param_ranges.keys())

        self.mode = self.config.SAMPLING["mode"].lower()
        self._geometry_mode_samplers: Dict[str, UnitSampler] = {}
        self._flow_mode_samplers: Dict[str, UnitSampler] = {}
        self._register_builtin_modes()

    # -------------------------
    # helpers
    # -------------------------
    def _register_builtin_modes(self) -> None:
        self.register_mode(
            "lhs",
            geometry_sampler=self._generate_lhs_samples,
            flow_sampler=self._generate_lhs_samples,
        )
        self.register_mode(
            "random",
            geometry_sampler=self._generate_random_samples,
            flow_sampler=self._generate_random_samples,
        )
        self.register_mode(
            "low_re_dense",
            geometry_sampler=self._generate_lhs_samples,
            flow_sampler=self._generate_low_re_dense_flow_samples,
        )

    def register_mode(
        self,
        mode: str,
        geometry_sampler: UnitSampler,
        flow_sampler: UnitSampler,
    ) -> None:
        normalized_mode = mode.lower()
        self._geometry_mode_samplers[normalized_mode] = geometry_sampler
        self._flow_mode_samplers[normalized_mode] = flow_sampler

    def _generate_unit_samples(
        self,
        n_params: int,
        n_samples: int,
        *,
        sampler_map: Dict[str, UnitSampler],
        sample_kind: str,
    ) -> np.ndarray:
        if self.mode not in sampler_map:
            supported_modes = sorted(set(sampler_map.keys()))
            raise ValueError(
                f"Unsupported {sample_kind} sampling mode '{self.mode}'. "
                f"Supported modes: {supported_modes}"
            )
        return sampler_map[self.mode](n_params, n_samples)

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

    def _generate_lhs_samples(
        self,
        n_params: int,
        n_samples: int,
    ) -> np.ndarray:
        return lhs(
            n_params,
            samples=n_samples,
            criterion=self.config.SAMPLING.get("lhs_criterion", "maximin"),
        )

    def _generate_random_samples(
        self,
        n_params: int,
        n_samples: int,
    ) -> np.ndarray:
        return np.random.random((n_samples, n_params))

    def _generate_low_re_dense_flow_samples(
        self,
        n_params: int,
        n_samples: int,
    ) -> np.ndarray:
        samples = lhs(
            n_params,
            samples=n_samples,
            criterion=self.config.SAMPLING.get("lhs_criterion", "maximin"),
        )

        try:
            re_idx = self.flow_param_names.index("reynolds_number")
        except ValueError:
            raise KeyError("FLOW_PARAM_RANGES must include 'reynolds_number'")

        alpha = self.config.SAMPLING.get("re_bias_alpha", 2.5)
        samples[:, re_idx] = samples[:, re_idx] ** alpha

        return samples

    # -------------------------
    # geometry sampling
    # -------------------------
    def generate_geometry_samples(self) -> np.ndarray:
        n_geom = self.config.SAMPLING["n_geometries"]
        n_geom_params = len(self.geom_param_ranges)

        geom_unit = self._generate_unit_samples(
            n_geom_params,
            n_geom,
            sampler_map=self._geometry_mode_samplers,
            sample_kind="geometry",
        )

        return self._scale_samples(
            geom_unit, self.geom_param_ranges, self.geom_param_names
        )

    # -------------------------
    # flow sampling
    # -------------------------
    def _generate_flow_samples_for_mode(self) -> np.ndarray:
        n_flow = self.config.SAMPLING["n_flow_per_geometry"]
        n_flow_params = len(self.flow_param_ranges)

        flow_unit = self._generate_unit_samples(
            n_flow_params,
            n_flow,
            sampler_map=self._flow_mode_samplers,
            sample_kind="flow",
        )

        return self._scale_samples(
            flow_unit, self.flow_param_ranges, self.flow_param_names
        )

    def generate_single_flow_sample_set(self, geom_idx: int) -> np.ndarray:
        _ = geom_idx  # keep signature for deterministic/per-geometry future strategies
        return self._generate_flow_samples_for_mode()

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

            mode = self.config.SAMPLING["mode"].lower()
            supported_modes = sorted(
                set(self._geometry_mode_samplers.keys()).intersection(
                    self._flow_mode_samplers.keys()
                )
            )
            if mode not in supported_modes:
                raise ValueError(f"Sampling mode must be one of: {supported_modes}")

            # validate ranges
            for param, (low, high) in self.geom_param_ranges.items():
                if low >= high:
                    raise ValueError(
                        f"Invalid geometry range for {param}: {low} >= {high}"
                    )

            for param, (low, high) in self.flow_param_ranges.items():
                if low >= high:
                    raise ValueError(f"Invalid flow range for {param}: {low} >= {high}")

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
