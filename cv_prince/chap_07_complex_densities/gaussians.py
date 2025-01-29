"""Contains scripts related to storing Gaussians"""

from dataclasses import dataclass
import numpy as np


@dataclass
class GaussParams:
    """Parameters used to generate a gaussian distribution"""

    mean: np.ndarray
    cov: np.ndarray
