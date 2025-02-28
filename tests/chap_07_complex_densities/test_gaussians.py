"""Test functions related to the gaussian"""

import pytest
import numpy as np
from scipy.stats import multivariate_normal
from cv_prince.chap_07_complex_densities.gaussians import Gaussian, GaussianParams


@pytest.mark.parametrize("gauss_params", [1, 2, 5, 10], indirect=True)
def test_pdf(gauss_params: GaussianParams, samples: np.ndarray):
    """Test if the custom Gaussian PDF generates same results as scipy"""

    # Generate the data
    gt_pdf = multivariate_normal(gauss_params.mean, gauss_params.cov).pdf(samples)

    # Our results
    gaussian = Gaussian(mean=gauss_params.mean, cov=gauss_params.cov)
    our_pdf = gaussian.pdf(samples)

    assert np.allclose(our_pdf, gt_pdf)
