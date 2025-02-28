"""Contains scripts related to storing Gaussians"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass
class GaussianParams:
    """Parameters used to define multivariate gaussian"""

    mean: np.ndarray
    cov: np.ndarray


class Gaussian:
    """Parameters used to generate a gaussian distribution"""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov

    @property
    def num_vars(self) -> int:
        """Number of variables in the Multivariate Gaussian"""
        return self.mean.shape[0]

    @cached_property
    def inv_cov(self) -> np.ndarray:
        """Compute and store inverse covariance matrix"""
        return np.linalg.inv(self.cov)

    @cached_property
    def normalization_factor(self) -> float:
        """Compute the probability normalisation factor"""

        two_pi_root = (2 * np.pi) ** (self.num_vars / 2)
        det_cov_root = np.sqrt(np.linalg.det(self.cov))

        return 1 / (two_pi_root * det_cov_root)

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

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        """Compute the probability of observing samples"""

        samples_centered = self.__center_samples(samples)  # (N, D)
        samples_qform = self.__quadratic_form(samples_centered)  # (N,)

        return self.normalization_factor * np.exp(-0.5 * samples_qform)

    def mahalanobis_dist(self, samples: np.ndarray):
        """Compute the Mahalanobis distance of samples to gaussian"""

        samples_centered = self.__center_samples(samples)  # (N, D)
        samples_qform = self.__quadratic_form(samples_centered)  # (N,)

        return np.sqrt(samples_qform)

    def __center_samples(self, samples: np.ndarray) -> np.ndarray:
        """Subtract mean from samples"""
        return samples - self.mean[np.newaxis, :]  # (N, D)

    def __quadratic_form(self, samples: np.ndarray) -> np.ndarray:
        """Compute x^T @ inv_cov @ x"""

        x_mu_sq = samples[:, np.newaxis, :] @ self.inv_cov[np.newaxis, ...]  # (N, 1, D)
        x_mu_sq = x_mu_sq @ samples[..., np.newaxis]  # (N, 1, 1)

        return x_mu_sq[..., 0, 0]  # (N,)
