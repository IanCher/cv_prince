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


@pytest.fixture(scope="module", name="num_vars")
def num_vars_fixture(request):
    """Number of variables for indirect parametrisation"""

    return request.param


@pytest.fixture(scope="module", name="num_components")
def num_components_fixture(request):
    """Number of components for indirect parametrisation"""

    return request.param


@pytest.fixture(scope="module", name="gmm_samples_num_components")
def gmm_samples_num_components_fixture(
    num_components: int, num_vars: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Sample points using a Gaussian Mixtuer Model"""

    num_samples = 1000

    mean = rng.uniform(-0.5, 0.5, (num_components, num_vars))

    cov = rng.uniform(-0.5, 0.5, (num_components, num_vars, 1))
    cov = cov @ cov.transpose((0, 2, 1)) + 0.1 * np.eye(num_vars)[None, ...]

    weights = rng.uniform(0.1, 0.8, (num_components,))
    weights /= weights.sum()

    selected_component = np.argmax(rng.multinomial(1, weights, (num_samples,)), axis=1)
    samples = np.empty((num_samples, num_vars), dtype=float)
    for k in range(num_components):
        samples_k = selected_component == k
        num_samples_k = samples_k.sum()
        samples[samples_k, :] = rng.multivariate_normal(
            mean=mean[k, ...], cov=cov[k, ...], size=(num_samples_k,)
        )

    return samples, num_components
