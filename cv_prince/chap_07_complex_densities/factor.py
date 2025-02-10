"""Contains scripts related to the generation of factor analysis"""

import numpy as np
from tqdm import tqdm


class ExpectationMaximisationFactor:

    def __init__(self, num_factors: int, seed: int | None = None):
        self.mean: np.ndarray | None = None
        self.diag_cov: np.ndarray | None = None
        self.factor_mtx: np.ndarray | None = None
        self.num_factors: int = num_factors

        self.dim: int | None = None

        self.__rng = np.random.default_rng(seed)  # RNG, can be initialised with seed
        self.__is_fitted: bool = False  # Indicates whether EM has been fit

    def fit(self, samples: np.ndarray, max_iter: int = 1000) -> None:
        """Fit a factor analysis on data using Expectation Maximisation

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        max_iter: int (default=1000)
            Max number of iterations if no convergence is reached
        """

        self.initialise_params(samples)

        obj_t0 = self.eval_objective(samples)
        for _ in tqdm(range(max_iter)):
            qh_mean, qh_cov = self.perform_e_step(samples)
            self.perform_m_step(samples, qh_mean, qh_cov)

            obj_t1 = self.eval_objective(samples)
            if obj_t1 - obj_t0 < 1e-7:
                break

            obj_t0 = obj_t1

    def initialise_params(self, samples: np.ndarray) -> None:
        num_samples = samples.shape[0]
        self.dim = samples.shape[1]

        assert (
            self.dim > self.num_factors
        ), f"Looking for {self.num_factors} factors, but there are only {self.dim} dimensions"

        # We estimate the maximum likelihood mean for a gaussian
        self.mean = 1 / num_samples * samples.sum(axis=0)
        self.diag_cov = np.ones((self.dim,))
        self.factor_mtx = np.eye(self.dim, self.num_factors)

    def perform_e_step(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qh_cov = self.factor_mtx.transpose() @ np.diag(1 / self.diag_cov)
        qh_cov = qh_cov @ self.factor_mtx
        qh_cov = qh_cov + np.eye(self.num_factors)
        qh_cov = np.linalg.inv(qh_cov)

        qh_mean = samples - self.mean[np.newaxis, :]  # (N, D)
        qh_mean = (
            np.diag(1 / self.diag_cov)[np.newaxis, ...] @ qh_mean[..., np.newaxis]
        )  # (N, D, 1)
        qh_mean = self.factor_mtx.transpose()[np.newaxis, ...] @ qh_mean  # (N, K, 1)
        qh_mean = qh_cov[np.newaxis, ...] @ qh_mean  # (N, K, 1)
        qh_mean = qh_mean[..., 0]  # (N, K)

        return qh_mean, qh_cov

    def perform_m_step(
        self, samples: np.ndarray, qh_mean: np.ndarray, qh_cov: np.ndarray
    ) -> None:
        num_samples = samples.shape[0]

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)

        x_mu_qh_mean = x_mu[..., np.newaxis] @ qh_mean[:, np.newaxis, :]  # (N, D, K)
        x_mu_qh_mean = x_mu_qh_mean.sum(axis=0)  # (D, K)

        qh_mean_sq = qh_mean[..., np.newaxis] @ qh_mean[:, np.newaxis, :]  # (N, K, K)
        exp_hht = num_samples * qh_cov + qh_mean_sq.sum(axis=0)  # (K, K)

        self.factor_mtx = np.linalg.lstsq(exp_hht, x_mu_qh_mean.transpose())[0]
        self.factor_mtx = self.factor_mtx.transpose()

        x_mu_sq = x_mu[:, :, np.newaxis] @ x_mu[:, np.newaxis, :]  # (N, D, D)
        qh_mean_x = qh_mean[..., np.newaxis] @ x_mu[:, np.newaxis, :]  # (N, K, D)
        phi_qh_mean_x = self.factor_mtx[np.newaxis, ...] @ qh_mean_x  # (N, D, D)

        inter_cov = x_mu_sq - phi_qh_mean_x
        inter_cov = inter_cov.sum(axis=0)
        self.diag_cov = 1 / num_samples * np.diag(inter_cov)

    def eval_objective(self, samples: np.ndarray) -> float:
        full_cov = self.factor_mtx @ self.factor_mtx.transpose() + np.diag(
            self.diag_cov
        )
        det_full_cov = np.linalg.det(full_cov)
        log_det_full_cov = np.log(det_full_cov)

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)
        x_mu_sq = (
            x_mu[:, np.newaxis, :] @ np.linalg.inv(full_cov)[np.newaxis, ...]
        )  # (N, 1, D)
        x_mu_sq = x_mu_sq @ x_mu[..., np.newaxis]  # (N, 1, 1)
        x_mu_sq = x_mu_sq[:, 0, 0]  # (N,)

        num_samples = samples.shape[0]

        return -1 / 2 * (num_samples * log_det_full_cov + x_mu_sq.sum())

    @property
    def rng(self) -> np.random.Generator:
        return self.__rng

    @property
    def is_fitted(self) -> bool:
        return self.__is_fitted

    def __str__(self):
        with np.printoptions(precision=3):
            return (
                "Factor analysis with the following parameters \n"
                f"* mean: \n{self.mean} \n\n"
                f"* diag_cov: \n{self.diag_cov} \n\n"
                f"* factor_mtx: \n{self.factor_mtx} \n"
            )
