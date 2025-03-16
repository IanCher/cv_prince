"""Contains scripts related to storing Gaussians"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianParams:
    """Parameters used to define multivariate gaussian"""

    mean: np.ndarray
    cov: np.ndarray


class Gaussian:
    """Parameters used to generate a gaussian distribution"""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.cov_cholesky_decomp = np.zeros_like(cov)
        self.cov_cholesky_inv = np.zeros_like(cov)
        self.det_cov_cholesky_decomp: float | None = 0.0
        self.log_normalization_factor: float | None = 0.0

        self.mean = mean
        self.cov = cov

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name == "cov":
            self.cov_cholesky_decomp = self.compute_cov_cholesky_decomp()
            self.cov_cholesky_inv = self.compute_cov_cholesky_inv()
            self.log_normalization_factor = self.compute_log_normalization_factor()
            self.det_cov_cholesky_decomp = self.compute_det_cov_cholesky_decomp()

    @property
    def num_vars(self) -> int:
        """Number of variables in the Multivariate Gaussian"""
        return self.mean.shape[0]

    @property
    def precision_mtx(self) -> np.ndarray:
        """Compute and store inverse covariance matrix"""

        return self.cov_cholesky_inv.T @ self.cov_cholesky_inv

    def compute_cov_cholesky_decomp(self) -> np.ndarray:
        """Compute the cholesky decomposition"""

        return np.linalg.cholesky(self.cov)

    def compute_cov_cholesky_inv(self) -> np.ndarray:
        """Compute the inverse of the cholesky decomposition"""
        return np.linalg.solve(self.cov_cholesky_decomp, np.eye(self.num_vars))

    def compute_log_normalization_factor(self) -> float:
        """Computer and store normalization factor for log pdf"""

        log_two_pi_root = -self.num_vars / 2 * np.log(2 * np.pi)
        log_det_cov_root = -np.log(np.diag(self.cov_cholesky_decomp)).sum()

        return log_two_pi_root + log_det_cov_root

    def compute_det_cov_cholesky_decomp(self) -> float:
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

        return np.exp(self.log_pdf(samples))

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

        quad_form = self.cov_cholesky_inv @ samples[..., np.newaxis]
        quad_form = np.square(quad_form[..., 0]).sum(axis=1)

        return quad_form  # (N,)
