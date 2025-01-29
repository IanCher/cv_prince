"""Contains scripts related to storing Gaussians"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Gaussian:
    """Parameters used to generate a gaussian distribution"""

    mean: np.ndarray
    cov: np.ndarray

    def sample(
        self,
        n: int = 1,
        seed: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray]:
        """Sample n points from the gaussian mixture model"""

        if rng is None:
            rng = np.random.default_rng(seed=seed)

        return rng.multivariate_normal(mean=self.mean, cov=self.cov, size=n)
