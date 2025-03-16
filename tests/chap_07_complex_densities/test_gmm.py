"""Test functions related to the Gausssian Mixture Model"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from cv_prince.chap_07_complex_densities.gaussians import Gaussian
from cv_prince.chap_07_complex_densities.gmm import GMMSampler


@pytest.mark.parametrize("num_vars", [1, 2, 5, 10], indirect=True)
@pytest.mark.parametrize("num_components", [1, 5, 10], indirect=True)
def test_gmm_sampler(gmm_samples_num_components: tuple[np.ndarray, int], seed: int):
    """Test GMM sampler log likelihood estimation"""
    samples, num_components = gmm_samples_num_components

    sklearn_gmm = GaussianMixture(n_components=num_components, random_state=seed)
    sklearn_gmm.fit(samples)
    gt_loglikelihood = sklearn_gmm.score_samples(samples)

    our_gmm = GMMSampler(
        sklearn_gmm.weights_,
        gaussians=[
            Gaussian(sklearn_gmm.means_[k, ...], cov=sklearn_gmm.covariances_[k, ...])
            for k in range(num_components)
        ],
    )
    our_loglikelihood = our_gmm.log_pdf(samples)

    assert np.allclose(gt_loglikelihood, our_loglikelihood)
