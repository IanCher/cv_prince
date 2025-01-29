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

    def mahalanobis_dist(self, samples: np.ndarray):
        """Compute the Mahalanobis distance of samples to gaussian"""

        inv_covs = np.linalg.inv(self.cov)  # (D, D)

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)
        x_mu_sq = x_mu[:, np.newaxis, :] @ inv_covs[np.newaxis, ...]  # (N, 1, D)
        x_mu_sq = x_mu_sq @ x_mu[..., np.newaxis]  # (N, 1, 1)
        x_mu_sq = x_mu_sq[..., 0, 0]  # (N,)

        return np.sqrt(x_mu_sq)
