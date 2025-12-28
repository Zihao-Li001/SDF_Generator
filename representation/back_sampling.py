from config import CONFIG

import numpy as np
from pyDOE2 import lhs
from typing import Tuple, List, Dict
import sys
import os
from .nonuniform_flow_sampler import NonUniformFlowSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParameterSampler:
    def __init__(self, config):
        self.config = config
        self.geom_param_ranges = config.GEOM_PARAM_RANGES
        self.flow_param_ranges = config.FLOW_PARAM_RANGES
        self.geom_param_names = list(self.geom_param_ranges.keys())
        self.flow_param_names = list(self.flow_param_ranges.keys())

        self.flow_sampler = NonUniformFlowSampler(
            n_samples=config.SAMPLING["n_flow_per_geometry"],
            angle_range=self.flow_param_ranges["incident_angle"],
            re_range=self.flow_param_ranges["reynolds_number"],
        )

    def _scale_samples(
        self,
        samples,
        param_ranges: Dict[str, Tuple[float, float]],
        param_names: list[str],
    ) -> np.ndarray:
        scaled = np.empty_like(samples)
        for i, param in enumerate(param_names):
            low, high = param_ranges[param]
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def _generate_lhs_samples(self, n_params, n_samples):
        return lhs(
            n_params, samples=n_samples, criterion=self.config.SAMPLING["lhs_criterion"]
        )

    def _generate_random_samples(self, n_params, n_samples):
        return np.random.random((n_samples, n_params))

    def generate_geometry_samples(self):
        n_geom = self.config.SAMPLING["n_geometries"]
        n_geom_params = len(self.geom_param_ranges)
        mode = self.config.SAMPLING["mode"]

        if mode == "lhs":
            geom_samples = self._generate_lhs_samples(n_geom_params, n_geom)

        elif mode == "physics":
            self.flow_sampler = NonUniformFlowSampler(
                n_samples=config.SAMPLING["n_flow_per_geometry"],
                angle_range=self.flow_param_ranges["incident_angle"],
                re_range=self.flow_param_ranges["reynolds_number"],
            )
        else:
            geom_samples = self._generate_random_samples(n_geom_params, n_geom)

        scaled_geom_samples = self._scale_samples(
            geom_samples, self.geom_param_ranges, self.geom_param_names
        )
        return scaled_geom_samples

    def _biased_logspace_from_unit(
        self, u: np.ndarray, low: float, high: float, alpha: float
    ) -> np.ndarray:
        log_low, log_high = np.log10(low), np.log10(high)
        u_biased = np.clip(u, 1e-12, 1 - 1e-12) ** alpha
        logs = log_low + (log_high - log_low) * u_biased
        return 10.0**logs

    def generate_single_flow_sample_set(self, geom_idx: int) -> np.ndarray:
        # n_flow = self.config.SAMPLING["n_flow_per_geometry"]
        # n_flow_params = len(self.flow_param_ranges)
        # mode = self.config.SAMPLING["mode"]
        #
        # if mode == "lhs":
        #     flow_samples = self._generate_lhs_samples(n_flow_params, n_flow)
        # else:
        #     flow_samples = self._generate_random_samples(n_flow_params, n_flow)
        #
        # re_idx = self.flow_param_names.index("reynolds_number")
        # re_low, re_high = self.flow_param_ranges["reynolds_number"]
        # u = flow_samples[:, re_idx]
        #
        # # bias number
        # alpha = 2.0
        # re_vals = self._biased_logspace_from_unit(u, re_low, re_high, alpha=alpha)
        #
        # re_normalized = (re_vals - re_low) / (re_high - re_low)
        # flow_samples[:, re_idx] = re_normalized
        #
        # scaled_flow_samples = self._scale_samples(
        #     flow_samples, self.flow_param_ranges, self.flow_param_names
        # )
        #
        # return scaled_flow_samples
        flow = self.flow_sampler.sample(geom_idx)
        out = np.zeros((flow.shape[0], len(self.flow_param_names)))
        for i, name in enumerate(self.flow_param_names):
            if name == "incident_angle":
                out[:, i] = flow[:, 0]
            elif name == "reynolds_number":
                out[:, i] = flow[:, 1]
            else:
                raise KeyError(f"Unsupported flow parameter: {name}")

        return out

    def generate_all_flow_samples(self, n_geometries: int):
        all_flow_samples = []

        for geom_idx in range(n_geometries):
            flow_samples = self.generate_single_flow_sample_set(geom_idx)
            all_flow_samples.append(flow_samples)
        return all_flow_samples

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
            "total_samples": self.config.total_samples,
            "sampling_mode": self.config.SAMPLING["mode"],
            "geometry_parameters": self.geom_param_names,
            "flow_parameters": self.flow_param_names,
            "geometry_ranges": self.geom_param_ranges,
            "flow_ranges": self.flow_param_ranges,
        }

    def validate_config(self):
        try:
            required_keys = ["n_geometries", "n_flow_per_geometry", "mode"]
            for key in required_keys:
                if key not in self.config.SAMPLING:
                    raise KeyError(f"Missing required_keys: {key}")
            if self.config.SAMPLING["n_geometries"] <= 0:
                raise ValueError("n_geometries must be postive")
            if self.config.SAMPLING["n_flow_per_geometry"] <= 0:
                raise ValueError("n_flow_per_geometry must be postive")
            if self.config.SAMPLING["mode"] not in ["lhs", "random"]:
                raise ValueError("Sampling mode must be 'lhs' or 'random'")
            for param, (low, high) in self.geom_param_ranges.items():
                if low >= high:
                    raise ValueError(f"Invalid range for {param}: {low} >= {high}")
            for param, (low, high) in self.flow_param_ranges.items():
                if low >= high:
                    raise ValueError(f"Invalid range for {param}: {low} >= {high}")
            return True
        except (KeyError, ValueError) as e:
            print(f"Configuration validation error: {e}")
            return False


if __name__ == "__main__":
    # This would normally come from your config
    config = CONFIG
    sampler = ParameterSampler(config)

    # Validate configuration
    if sampler.validate_config():
        print("Configuration is valid!")

        # Generate samples
        geom_samples, flow_samples_list = sampler.sample()

        print(f"Generated {len(geom_samples)} geometry samples")
        print(f"Generated flow samples for {len(flow_samples_list)} geometries")
        print(f"Each geometry has {len(flow_samples_list[0])} flow conditions")

        # Print sample info
        info = sampler.get_sample_info()
        print("\nSample Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("Configuration validation failed!")
