"""Contains kernel functions"""

import numpy as np


def radial_basis_func(
    array_1: np.ndarray,
    array_2: np.ndarray,
    lam: float,
    along_sample_axis: bool = False,
) -> np.ndarray:
    """RBF Kernel"""

    if len(array_1.shape) == 1:
        array_1 = np.expand_dims(array_1, 0)

    if len(array_2.shape) == 1:
        array_2 = np.expand_dims(array_2, 0)

    squared_norm_1 = np.einsum("kj, kj -> j", array_1, array_1)
    squared_norm_2 = np.einsum("kj, kj -> j", array_2, array_2)

    if along_sample_axis:
        dot_prod_12 = np.einsum("kj, kj->j", array_1, array_2)
    else:
        dot_prod_12 = np.einsum("ki, kj -> ij", array_1, array_2)
        squared_norm_1 = squared_norm_1[:, None]
        squared_norm_2 = squared_norm_2[None, :]

    quad_form_12 = squared_norm_1 + squared_norm_2 - 2 * dot_prod_12

    return np.exp(-quad_form_12 / (2 * lam * lam))
