"""This folder implements methods from chapter 8"""

# pylint: disable=c0103

from abc import ABC, abstractmethod

import numpy as np


class BaseRegression(ABC):
    """Base class for running regression methods"""

    @abstractmethod
    def fit(self, X: np.ndarray, w: np.ndarray):
        """Fit a regression model from data X and observation w"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts world state from data X"""

    @abstractmethod
    def log_likelihood(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Estimates log likelihood for data X and observation w"""

    def likelihood(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Estimates likelihood for data X and observation w"""

        return np.exp(self.log_likelihood(X, w))
