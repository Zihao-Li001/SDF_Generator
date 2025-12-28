import numpy as np


class NonUniformFlowSampler:
    """
    Non-uniform sampler for (incident_angle, reynolds_number)
    with Latin pairing, deterministic per geometry.
    """

    def __init__(
        self,
        n_samples: int = 32,
        angle_range=(0.0, 180.0),
        re_range=(10.0, 300.0),
        angle_jitter=1.0,
        re_jitter=0.05,
    ):
        self.n = n_samples
        self.angle_range = angle_range
        self.re_range = re_range
        self.angle_jitter = angle_jitter
        self.re_jitter = re_jitter

        # 非均匀分配（与你前面确认的一致）
        self.angle_bins = [
            (0.0, 30.0, 12),
            (30.0, 90.0, 10),
            (90.0, 180.0, 10),
        ]
        self.re_bins = [
            (10.0, 50.0, 14),
            (50.0, 150.0, 10),
            (150.0, 300.0, 8),
        ]

    def _sample_angles(self, rng: np.random.Generator) -> np.ndarray:
        angles = []
        for lo, hi, k in self.angle_bins:
            a = np.linspace(lo, hi, k, endpoint=False)
            angles.append(a)
        angles = np.concatenate(angles)

        angles += rng.uniform(-self.angle_jitter, self.angle_jitter, size=self.n)
        return np.clip(angles, *self.angle_range)

    def _sample_res(self, rng: np.random.Generator) -> np.ndarray:
        res = []
        for lo, hi, k in self.re_bins:
            r = np.logspace(np.log10(lo), np.log10(hi), k, endpoint=False)
            res.append(r)
        res = np.concatenate(res)

        res *= rng.uniform(1.0 - self.re_jitter, 1.0 + self.re_jitter, size=self.n)
        return np.clip(res, *self.re_range)

    def sample(self, geom_idx: int) -> np.ndarray:
        """
        Return shape: (32, 2) -> [incident_angle, reynolds_number]
        """
        rng = np.random.default_rng(seed=geom_idx)

        angles = self._sample_angles(rng)
        res = self._sample_res(rng)

        perm = rng.permutation(self.n)
        paired = np.stack([angles, res[perm]], axis=1)
        return paired
