"""Implementation of Paragraph 8.6"""

# pylint: disable=c0103

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cv_prince.chap_08_linear_regression import BaseRegression


@dataclass
class SparseLinRegParams:
    """Parameters for running relevance vector regression"""

    nu: float
    thresh: float
    niter: int = 1000


class SparseLinearRegression(BaseRegression):
    """Sparse Linear Regression using Student t distribution prior"""

    def __init__(self, params: SparseLinRegParams):
        self.params = params
        self.hidden_vars: np.ndarray | None = None
        self.sigma: float | None = None
        self.post_cov: np.ndarray | None = None
        self.post_mean: np.ndarray | None = None
        self.tokeep: np.ndarray | None = None

    def fit(self, X, w):

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
        ndims, nsamples = X.shape

        wTw = np.dot(w, w)
        Xw = X @ w
        XXT = X @ X.T
        self.hidden_vars = np.ones((ndims,), dtype=float)

        def sigma_obj_func(sigma):
            cov = X.T @ np.diag(1 / self.hidden_vars) @ X + sigma * np.eye(nsamples)

            log_det_cov = np.log(np.linalg.det(cov))

            woodburry_mtx = 1 / sigma * XXT + np.diag(self.hidden_vars)
            woodburry_quad = np.linalg.solve(woodburry_mtx, Xw)
            woodburry_quad = np.dot(Xw, woodburry_quad)

            log_exp = 1 / sigma * (wTw - 1 / sigma * woodburry_quad)

            return log_det_cov + log_exp

        for _ in tqdm(range(1000)):
            self.sigma = minimize_scalar(sigma_obj_func, bounds=[1e-6, 1e6]).x

            A = 1 / self.sigma * XXT + np.diag(self.hidden_vars)
            self.post_cov = np.linalg.inv(A)
            self.post_mean = 1 / self.sigma * self.post_cov @ Xw

            hidden_prev = self.hidden_vars.copy()
            self.hidden_vars = 1 - self.hidden_vars * self.post_cov.flat[:: ndims + 1]
            self.hidden_vars += self.nu

            self.hidden_vars /= self.post_mean * self.post_mean + self.nu

            if np.allclose(hidden_prev, self.hidden_vars):
                break

        self.tokeep = np.concatenate([self.hidden_vars[:-1] < self.thresh, [True]])
        self.post_mean = self.post_mean[self.tokeep]
        self.post_cov = self.post_cov[:, self.tokeep][self.tokeep, :]

    def predict(self, X):

        if self.post_mean is None:
            raise ValueError("Linear regression model has not been fitted")

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
        X = X[self.tokeep, :]

        return self.post_mean @ X

    def log_likelihood(self, X, w):

        if self.post_mean is None:
            raise ValueError("Linear regression model has not been fitted")

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
        X = X[self.tokeep, :]

        estimated_cov = np.sum(X * (self.post_cov @ X), axis=0)
        estimated_cov += self.sigma

        log_likelihood = np.log(2 * np.pi * estimated_cov)
        log_likelihood += 1 / estimated_cov * (w - self.post_mean @ X) ** 2
        log_likelihood *= -0.5

        return log_likelihood

    @property
    def nu(self) -> float:  # pylint: disable=c0116
        return self.params.nu

    @property
    def thresh(self) -> float:  # pylint: disable=c0116
        return self.params.thresh

    @property
    def niter(self) -> int:  # pylint: disable=c0116
        return self.params.niter
