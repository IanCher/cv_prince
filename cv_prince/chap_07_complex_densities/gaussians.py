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
    def precision_mtx(self) -> np.ndarray:
        """Compute and store inverse covariance matrix"""

        return self.cov_cholesky_inv.T @ self.cov_cholesky_inv

    @cached_property
    def cov_cholesky_decomp(self) -> np.ndarray:
        """Compute the cholesky decomposition"""
        return np.linalg.cholesky(self.cov)

    @cached_property
    def cov_cholesky_inv(self) -> np.ndarray:
        """Compute the inverse of the cholesky decomposition"""
        return np.linalg.solve(self.cov_cholesky_decomp, np.eye(self.num_vars))

    @cached_property
    def normalization_factor(self) -> float:
        """Compute the probability normalisation factor"""

        two_pi_root = (2 * np.pi) ** (self.num_vars / 2)

        return 1 / (two_pi_root * self.det_cov_cholesky_decomp)

    @cached_property
    def log_normalization_factor(self) -> float:
        """Computer and store normalization factor for log pdf"""

        log_two_pi_root = -self.num_vars / 2 * np.log(2 * np.pi)
        log_det_cov_root = -np.log(np.diag(self.cov_cholesky_decomp)).sum()

        return log_two_pi_root + log_det_cov_root

    @cached_property
    def det_cov_cholesky_decomp(self) -> float:
        """Compute the determinant of the covariance matrix"""
        return np.prod(np.diag(self.cov_cholesky_decomp))

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

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        """Compute the log probability of observed samples"""

        samples_centered = self.__center_samples(samples)  # (N, D)
        samples_qform = self.__quadratic_form(samples_centered)  # (N,)

        return self.log_normalization_factor - 1 / 2 * samples_qform

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

        quad_form = self.cov_cholesky_inv[np.newaxis, ...] @ samples[..., np.newaxis]
        quad_form = self.cov_cholesky_inv.T[np.newaxis, ...] @ quad_form
        quad_form = samples[:, np.newaxis, :] @ quad_form

        return quad_form[..., 0, 0]  # (N,)
