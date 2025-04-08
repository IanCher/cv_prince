"""Implementation of paragraph 8.8"""

# pylint: disable=c0103

from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, Path(__file__).parents[2])

from cv_prince.chap_08_linear_regression import BaseRegression
from cv_prince.chap_08_linear_regression.kernels import radial_basis_func


class RelevanceVectorRegression(BaseRegression):
    """Fits and applies relevance vector regression model
    (non linear sparse bayesian regression with RBF)"""

    def __init__(self, nu: float, lam: float, thresh: float, niter: int = 1000):
        self.nu = nu
        self.lam = lam
        self.thresh = thresh
        self.niter = niter
        self.hidden_vars: np.ndarray | None = None
        self.sigma: float | None = None
        self.intermediate_post_cov: np.ndarray | None = None
        self.intermediate_post_mean: np.ndarray | None = None
        self.tokeep: np.ndarray | None = None
        self.training_data: np.ndarray | None = None
        self.training_obs: np.ndarray | None = None

    def fit(self, X, w):

        if len(X.shape) == 1:
            X = np.expand_dims(X, 0)

        KXX = self.__kernel(X, X)
        KXX_sq = KXX @ KXX
        KXX_w = KXX @ w

        def sigma_obj_func(sigma):
            M = KXX @ (1 / self.hidden_vars[:, None] * KXX) + sigma * np.eye(
                KXX.shape[0]
            )

            log_det_M = np.log(np.linalg.det(M))

            M_inv_w = np.linalg.solve(M, w)
            w_M_inv_w = np.dot(w, M_inv_w)

            return log_det_M + w_M_inv_w

        self.hidden_vars = np.ones((X.shape[1],))

        for _ in range(self.niter):
            self.sigma = minimize_scalar(sigma_obj_func, bounds=[1e-6, 1e6]).x

            A = 1 / self.sigma * KXX_sq + np.diag(self.hidden_vars)
            self.intermediate_post_cov = np.linalg.inv(A)
            self.intermediate_post_mean = self.intermediate_post_cov @ KXX_w
            self.intermediate_post_mean /= self.sigma

            hprev = self.hidden_vars.copy()
            self.hidden_vars *= -self.intermediate_post_cov.flat[:: X.shape[1] + 1]
            self.hidden_vars += 1 + self.nu
            self.hidden_vars /= self.intermediate_post_mean**2 + self.nu

            if np.allclose(hprev, self.hidden_vars):
                break

        self.tokeep = self.hidden_vars < self.thresh
        self.training_data = X[:, self.tokeep]
        self.training_obs = w[self.tokeep]

        KXX_red = self.__kernel(self.training_data, self.training_data)
        KXX_sq_red = KXX_red @ KXX_red

        A_red = 1 / self.sigma * KXX_sq_red + np.diag(self.hidden_vars[self.tokeep])

        self.intermediate_post_mean = KXX_red @ w[self.tokeep]
        self.intermediate_post_cov = np.linalg.inv(A_red)

    def predict(self, X):
        KXx = self.__kernel(self.training_data, X)

        bayes_mean = KXx.T @ (self.intermediate_post_cov @ self.intermediate_post_mean)
        bayes_mean /= self.sigma

        return bayes_mean

    def log_likelihood(self, X, w):
        KXx = self.__kernel(self.training_data, X)

        bayes_mean = KXx.T @ (self.intermediate_post_cov @ self.intermediate_post_mean)
        bayes_mean /= self.sigma

        bayes_var = np.sum(KXx * (self.intermediate_post_cov @ KXx), axis=0)
        bayes_var += self.sigma

        log_likelihood = np.log(2 * np.pi * bayes_var)
        log_likelihood += (w - bayes_mean) ** 2 / bayes_var
        log_likelihood *= -0.5

        return log_likelihood

    def __kernel(
        self, array_1: np.ndarray, array_2: np.ndarray, along_sample_axis: bool = False
    ) -> np.ndarray:
        return radial_basis_func(array_1, array_2, self.lam, along_sample_axis)
