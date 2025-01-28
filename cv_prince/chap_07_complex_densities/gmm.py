"""Contains scripts related to the generation of gaussian mixture models"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np
from tqdm import tqdm


@dataclass
class GaussParams:
    """Parameters used to generate multivariate gaussian distribution"""

    mean: np.ndarray
    cov: np.ndarray


@dataclass
class GMMSampler:
    """Object used to sample from gaussian mixture models

    Parameters
    ----------
    weights: tuple[float]
        The weights for each gaussian component of the GMM. Must sum to 1.
    gaussian_params: tuple[GaussParams]
        The gaussian parameters associated with each component. Must have the same
        length as weights, such that

        P(x) = sum weights[i] * Norm(mean=gauss_params[i].mean, cov=gauss_params[i].cov)
    """

    weights: tuple[float]
    gaussians_params: tuple[GaussParams]

    def __post_init__(self):
        assert len(self.weights) == len(self.gaussians_params), (
            f"{len(self.weights)} weights were provided, "
            f"but {len(self.gaussians_params)} gaussians parameters set."
        )

        if not np.isclose(sum(self.weights), 1):
            raise ValueError("The weights of the GMM components must sum to 1.")

    def sample_points(self, n: int = 1, seed: float | None = None) -> tuple[np.ndarray]:
        """Sample n points from the gaussian mixture model"""

        rng = np.random.default_rng(seed=seed)

        hidden_samples = rng.multinomial(n, self.weights)

        return [
            rng.multivariate_normal(
                mean=gaussian_params.mean,
                cov=gaussian_params.cov,
                size=ncomponent_samples,
            )
            for (ncomponent_samples, gaussian_params) in zip(
                hidden_samples, self.gaussians_params
            )
        ]


class ExpectationMaximisationGMM:
    """Object performing Expectation Maximisation in order to fit a GMM"""

    def __init__(self, num_components: int, seed: int | None = None):
        self.num_components = num_components
        self.weights: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None

        self.ndims: int | None = None

        self.__rng = np.random.default_rng(seed)
        self.__is_fitted: bool = False

    def fit(self, samples: np.ndarray, max_iter: int = 1000) -> None:
        """Estimate the parameters of the GMM from samples

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension
        max_iter: int
            Maximum number of iterations if convergence is not reached
        """
        self.initialise_params(samples)

        log_likelihood_t0 = self.compute_log_likelihood(samples)

        for _ in tqdm(range(max_iter)):
            posteriors = self.perform_e_step(samples)
            self.perform_m_step(samples, posteriors)

            log_likelihood_t1 = self.compute_log_likelihood(samples)

            if log_likelihood_t1 - log_likelihood_t0 < 1e-7:
                break

            log_likelihood_t0 = log_likelihood_t1

        self.__is_fitted = True

    def compute_log_likelihood(self, samples: np.ndarray) -> float:
        """Compute the log likelihood of the sample with the current GMM estimates

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        log likelihood: float
            Computed as sum_i logP(x_i)
        """

        join_probs = self.compute_join_prob(samples)
        evidences = np.sum(join_probs, axis=-1)  # (N,)

        return np.log(evidences).sum()

    def perform_e_step(self, samples: np.ndarray) -> np.ndarray:
        """Performs the E-Step in the expectation maximisation algorithm
        It consists in computing the posterior over the hidden variables


        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        posteriors: np.ndarray
            P(h_i | x_i, theta^(t)), posteriors over all samples and components.
            Shape (N, K) where N is the number of samples and K the number of components
        """

        join_probs = self.compute_join_prob(samples)
        evidences = np.sum(join_probs, axis=-1, keepdims=True)  # (N, 1)
        posteriors = join_probs / evidences  # (N, K)

        return posteriors

    def perform_m_step(self, samples, posteriors) -> None:
        """Performs the E-Step in the expectation maximisation algorithm
        We maximise the E[log P(x, h)] over the GMM parameters for the posteriors

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension
        posteriors: np.ndarray
            P(h_i | x_i, theta^(t)), posteriors computed at the E-step.
            Shape (N, K) where N is the number of samples and K the number of components

        """
        scaled_w = posteriors.sum(axis=0)  # (K,)

        self.update_weights(scaled_w)
        self.update_means(samples, posteriors, scaled_w)
        self.update_covs(samples, posteriors, scaled_w)

    def compute_join_prob(self, samples: np.ndarray) -> np.ndarray:
        """Compute the join probability P(x_i, h_i) = P(x_i|h_i) * P(hi)

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        join_prob: np.ndarray
            Shape (N, K) indicating the join probability for each pair (x_i, h_i)
        """

        det_covs_root = np.sqrt(np.linalg.det(self.covs))  # (K,)
        normaliser = self.two_pi_root * det_covs_root[np.newaxis, ...]  # (1, K)

        inv_covs = np.linalg.inv(self.covs)  # (K, D, D)

        x_mu = samples[:, np.newaxis, ...] - self.means[np.newaxis, :, :]  # (N, 1, D)
        x_mu_sq = x_mu[:, :, np.newaxis, :] @ inv_covs[np.newaxis, ...]  # (N, K, 1, D)
        x_mu_sq = x_mu_sq @ x_mu[:, :, :, np.newaxis]  # (N, K, 1, 1)
        x_mu_sq = x_mu_sq[..., 0, 0]  # (N, K)

        likelihoods = 1 / normaliser * np.exp(-0.5 * x_mu_sq)  # (N, K)
        priors = self.weights[np.newaxis, :]  # (1, K)

        return likelihoods * priors  # (N, K)

    def update_weights(self, scaled_w: np.ndarray) -> None:
        """Update the GMM weights using the Maximum Likelihood formula"""

        self.weights = scaled_w / scaled_w.sum()

    def update_means(
        self, samples: np.ndarray, posteriors: np.ndarray, scaled_w: np.ndarray
    ) -> None:
        """Update the GMM means using the Maximum Likelihood formula"""

        weighted_samples = (
            samples[:, np.newaxis, :] * posteriors[..., np.newaxis]
        )  # (N, K, D)
        self.means = weighted_samples.sum(axis=0) / scaled_w[:, np.newaxis]

    def update_covs(
        self, samples: np.ndarray, posteriors: np.ndarray, scaled_w: np.ndarray
    ) -> None:
        """Update the GMM covariances using the Maximum Likelihood formula"""

        x_mu = samples[:, np.newaxis, :] - self.means[np.newaxis, :, :]  # (N, K, D)
        x_mu_sq = x_mu[..., np.newaxis] @ x_mu[:, :, np.newaxis, :]  # (N, K, D, D)

        weighted_x_mu_sq = (
            x_mu_sq * posteriors[..., np.newaxis, np.newaxis]
        )  # (N, K, D, D)

        self.covs = weighted_x_mu_sq.sum(axis=0) / scaled_w[:, np.newaxis, np.newaxis]

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """For each sample predicts which component is most likely to have generated it

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        label: np.ndarray
            Integer array of shape (N,) with taking values in {0, 1, ..., K-1}
            where K is the number Gaussian components
        """
        if not self.is_fitted:
            raise ValueError("GMM has not been fitted, cannot make predictions.")

        join_probs = self.compute_join_prob(samples)  # (N, K)

        return np.argmax(join_probs, axis=1)

    def initialise_params(self, samples: np.ndarray) -> None:
        """Naive initialisation for the parameters
        We take K random samples to serve as initial means.
        We initialise the covariances as isotropic covariance matrics with the weight
        being a fraction of the samples extent.

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension
        """
        self.ndims = samples.shape[1]

        self.weights = np.full((self.num_components,), 1.0 / self.num_components)

        random_samples = self.rng.choice(
            np.arange(samples.shape[0]), (self.num_components,), replace=False
        )
        self.means = samples[random_samples, :]

        samples_extent = samples.max(axis=0) - samples.min(axis=0)
        cov_init = np.diag(0.1 * samples_extent)
        self.covs = np.stack(self.num_components * [cov_init], axis=0)

    @cached_property
    def two_pi_root(self) -> float:
        """Stores (2 * np.pi) ** (self.ndims / 2) to avoid recomputing it"""
        if self.ndims is None:
            raise ValueError("The number of dimensions has not been defined")

        return (2 * np.pi) ** (self.ndims / 2)

    @property
    def rng(self) -> np.random.Generator:
        """Access the random generator"""
        return self.__rng

    @property
    def is_fitted(self) -> bool:
        """Checks whether the model has been fitted or not"""
        return self.__is_fitted

    def __str__(self):
        with np.printoptions(precision=3):
            return (
                "GMM with the following parameters \n"
                f"* weights: \n{self.weights} \n\n"
                f"* means  : \n{self.means} \n\n"
                f"* covs   : \n{self.covs}"
            )
