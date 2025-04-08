"""Implementations of paragraph 8.3"""

# pylint: disable=c0103
from pathlib import Path
import sys
import numpy as np

# pylint: disable=wrong-import-position
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cv_prince.chap_08_linear_regression import BaseRegression


class NonLinearRegression(BaseRegression):
    """
    Object to fit and manipulate non linear regression models in 1D
    Currently implements the Radial Basis function
    """

    def __init__(self, alphas: np.ndarray, lam: float):
        self.alphas = alphas
        self.lam = lam
        self.phi: np.ndarray | None = None
        self.sigma: float | None = None

    def fit(self, X: np.ndarray, w: np.ndarray):
        """Fits a non linear regression model"""

        Z = self.__non_lin_func(X).T
        ZZT = Z @ Z.T
        Zw = Z @ w

        ZZT_chol = np.linalg.cholesky(ZZT, upper=False)
        self.phi = np.linalg.solve(ZZT_chol, Zw)
        self.phi = np.linalg.solve(ZZT_chol.T, self.phi)

        diff = w - self.phi @ Z
        quad = np.dot(diff, diff)
        self.sigma = 1 / len(w) * quad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the world state w from the observed data X"""

        if self.phi is None:
            raise ValueError("Linear regression model has not been fitted")

        return self.__non_lin_func(X) @ self.phi

    def log_likelihood(self, X: np.ndarray, w: np.ndarray):
        """Estimates the log likelihood for data X and world state w"""

        if self.phi is None:
            raise ValueError("Linear regression model has not been fitted")

        Z = self.__non_lin_func(X)

        log_likelihood = np.log(2 * np.pi * self.sigma)
        log_likelihood += (w - Z @ self.phi) ** 2 / self.sigma
        log_likelihood *= -0.5

        return log_likelihood

    def __non_lin_func(self, X: np.ndarray) -> np.ndarray:
        """Applies a non linear function to the collumns of X"""

        return np.stack(
            [np.exp(-((X - alpha) ** 2) / self.lam) for alpha in self.alphas], axis=-1
        )
