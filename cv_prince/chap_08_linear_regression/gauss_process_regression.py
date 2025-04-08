"""Implements paragraph 8.5"""

# pylint: disable=c0103

from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cv_prince.chap_08_linear_regression import BaseRegression
from cv_prince.chap_08_linear_regression.kernels import radial_basis_func


class GaussProcessRegression(BaseRegression):
    """Bayesian Non linear regression using rbf kernel"""

    def __init__(self, lam: float, sigma_prior: float):
        self.lam = lam
        self.sigma_prior = sigma_prior
        self.sigma: np.ndarray | None = None
        self.intermediate_post_cov: np.ndarray | None = None
        self.intermediate_post_mean: np.ndarray | None = None
        self.training_data: np.ndarray | None = None
        self.training_obs: np.ndarray | None = None

    def fit(self, X: np.ndarray, w: np.ndarray):
        self.training_data = X
        self.training_obs = w

        KXX = self.__kernel(X, X)
        KXXw = KXX @ w

        def sigma_obj_func(sigma: float):
            """Objective function to optimise sigma"""

            M = self.sigma_prior * KXX + sigma * np.eye(KXX.shape[0])
            M_chol = np.linalg.cholesky(M)

            log_det_M = 2 * np.log(M_chol.diagonal()).sum()

            M_inv_w = np.linalg.solve(M_chol, w)
            M_inv_w = np.linalg.solve(M_chol.T, M_inv_w)
            wT_M_in_w = w @ M_inv_w

            return log_det_M + wT_M_in_w

        self.sigma = minimize_scalar(sigma_obj_func, bounds=[1e-6, 1e6]).x

        M = KXX + self.sigma / self.sigma_prior * np.eye(KXX.shape[0])
        self.intermediate_post_cov = np.linalg.inv(M)
        self.intermediate_post_mean = self.intermediate_post_cov @ KXXw

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.intermediate_post_mean is None:
            raise ValueError("Linear regression model has not been fitted")

        KXx = self.__kernel(self.training_data, X)

        mean = KXx.T @ self.training_obs - self.intermediate_post_mean @ KXx
        mean *= self.sigma_prior / self.sigma

        return mean

    def log_likelihood(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:

        if self.intermediate_post_mean is None:
            raise ValueError("Linear regression model has not been fitted")

        KXx = self.__kernel(self.training_data, X)

        mean = KXx.T @ self.training_obs - self.intermediate_post_mean @ KXx
        mean *= self.sigma_prior / self.sigma

        cov = self.__kernel(X, X, along_sample_axis=True)
        cov -= np.sum(KXx * (self.intermediate_post_cov @ KXx), axis=0)
        cov *= self.sigma_prior
        cov += self.sigma

        return -0.5 * (np.log(2 * np.pi * cov) + (w - mean) ** 2 / cov)

    def __kernel(
        self, array_1: np.ndarray, array_2: np.ndarray, along_sample_axis: bool = False
    ) -> np.ndarray:
        return radial_basis_func(array_1, array_2, self.lam, along_sample_axis)
