"""Configuration for the tests concerning chap07"""

import numpy as np
import pytest

from cv_prince.chap_07_complex_densities.gaussians import GaussianParams


@pytest.fixture(scope="module", name="gauss_params")
def gauss_params_fixture(request, rng: np.random.Generator) -> np.ndarray:
    """Random mean"""
    num_vars = request.param

    mean = rng.uniform(-0.5, 0.5, (num_vars,))

    cov = rng.uniform(-0.5, 0.5, (num_vars,))
    cov = cov @ cov.T + 0.1 * np.eye(num_vars)

    return GaussianParams(mean=mean, cov=cov)


@pytest.fixture(scope="module", name="samples")
def samples_fixture(
    gauss_params: GaussianParams, rng: np.random.Generator
) -> np.ndarray:
    """Random Samples obtained from mean and cov"""
    return rng.multivariate_normal(
        mean=gauss_params.mean, cov=gauss_params.cov, size=(10,)
    )
