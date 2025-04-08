"""All things related to bayesian linear regression"""

# pylint: disable=c0103

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cv_prince.chap_08_linear_regression import BaseRegression


class BayesLinearRegression(BaseRegression):
    """Fits a bayesian linear regression models and uses it"""

    def __init__(self, sigma_prior: float):
        self.sigma_prior = sigma_prior
        self.sigma: np.ndarray | None = None
        self.post_cov: np.ndarray | None = None
        self.post_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, w: np.ndarray):
        """Fits a bayesian linear regression model"""

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        ndims, nsamples = X.shape
        XXT = X @ X.T
        Xw = X @ w
        wTw = np.dot(w, w)

        def sigma_obj_func(sigma: float):
            """Objective function to optimise sigma"""
            # We use the matrix determinant lemma (assume ndims < nsamples)

            intermediate_mtx = XXT + sigma / self.sigma_prior * np.eye(ndims)
            intermediate_mtx_chol = np.linalg.cholesky(intermediate_mtx, upper=False)

            log_det = 2 * np.sum(np.log(np.diag(intermediate_mtx_chol)))
            log_det += (nsamples - ndims) * np.log(sigma)

            # Compute (XXT + sigma / sigma_p * I_D)^(-1) * X * w
            quad_form = np.linalg.solve(intermediate_mtx_chol, Xw)
            quad_form = np.linalg.solve(intermediate_mtx_chol.T, quad_form)

            # Compute w^T * X^T * (1 / sigma * XXT + 1 / sigma_p * I_D)^(-1) * X * w
            quad_form = np.sum(Xw * quad_form)

            # Complete Woodburry identity to have full quadratic form wT * M^(-1) * w
            quad_form = wTw - quad_form
            quad_form /= sigma

            return log_det + quad_form

        self.sigma = minimize_scalar(sigma_obj_func, bounds=[1e-6, 1e6]).x

        intermediate_mtx = 1 / self.sigma_prior * np.eye(ndims) + 1 / self.sigma * XXT
        self.post_cov = np.linalg.inv(intermediate_mtx)
        self.post_mean = 1 / self.sigma * self.post_cov @ Xw

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts world state from data X using bayesian linear regression model"""

        if self.post_mean is None:
            raise ValueError("Linear regression model has not been fitted")

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        return self.post_mean @ X

    def log_likelihood(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Estimates the log likelihood for data X and world state w"""

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        estimated_cov = np.sum(X * (self.post_cov @ X), axis=0)
        estimated_cov += self.sigma

        log_likelihood = np.log(2 * np.pi * estimated_cov)
        log_likelihood += 1 / estimated_cov * (w - self.post_mean @ X) ** 2
        log_likelihood *= -0.5

        return log_likelihood
