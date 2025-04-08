"""All methods related to linear regression using maximum likelihood"""

# pylint: disable=c0103
from pathlib import Path
import sys
import numpy as np

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cv_prince.chap_08_linear_regression import BaseRegression


class LinearRegression(BaseRegression):
    """Performs linear regression using maximum likelihood"""

    def __init__(self):
        self.phi: np.ndarray | None = None
        self.sigma: float | None = None

    def fit(self, X: np.ndarray, w: np.ndarray):
        """Estimates the parameters of the linear regression model"""

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        nsamples = X.shape[1]
        X = np.concatenate([X, np.ones((1, nsamples))], axis=0)

        self.phi = np.linalg.solve(X @ X.T, X @ w)

        diff = w - X.T @ self.phi
        diff_sq = np.sum(diff * diff)

        self.sigma = 1 / nsamples * diff_sq

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the world state w using the linear regression model"""

        if self.phi is None:
            raise ValueError("Linear regression model has not been fitted")

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        nsamples = X.shape[1]
        X = np.concatenate([X, np.ones((1, nsamples))], axis=0)

        return self.phi @ X

    def log_likelihood(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Estimate the log likelihood for data X and world state w"""

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        nsamples = X.shape[1]
        X = np.concatenate([X, np.ones((1, nsamples))], axis=0)

        log_likelihood = np.log(2 * np.pi * self.sigma)
        log_likelihood += (w - X.T @ self.phi) ** 2 / self.sigma
        log_likelihood *= -0.5

        return log_likelihood
