"""Contains scripts related to the generation of gaussian mixture models"""

from dataclasses import dataclass
from functools import cached_property
from typing import Generator
from warnings import warn
import numpy as np
from tqdm import tqdm
from .gaussians import Gaussian


@dataclass
class GMMParams:
    """Gaussian Mixture Model parameters"""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray


class GMMSampler:
    """Object used to sample from gaussian mixture models

    Parameters
    ----------
    weights: tuple[float]
        The weights for each gaussian component of the GMM. Must sum to 1.
    gaussians: tuple[Gaussians]
        The gaussian parameters associated with each component. Must have the same
        length as weights, such that

        P(x) = sum_i weights[i] * Norm(mean=gaussians[i].mean, cov=gaussians[i].cov)
    """

    def __init__(self, weights: tuple[float], gaussians: tuple[Gaussian]):

        self.weights = weights
        self.gaussians = gaussians

        assert len(self.weights) == len(self.gaussians), (
            f"{len(self.weights)} weights were provided, "
            f"but {len(self.gaussians)} gaussians parameters set."
        )

        if not np.isclose(sum(self.weights), 1):
            warn("The weights of the GMM components must sum to 1.")

    @property
    def num_components(self) -> int:
        """Number of gaussian components"""

        return len(self.weights)

    @property
    def num_vars(self) -> int:
        """Number of variables in each MVG component"""

        return self.gaussians[0].num_vars

    @property
    def components(self) -> Generator[tuple[float, Gaussian], None, None]:
        """Iterates over the components"""

        for i in range(self.num_components):
            yield (self.weights[i], self.gaussians[i])

    def sample(
        self,
        n: int = 1,
        seed: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray]:
        """Sample n points from the gaussian mixture model"""

        if rng is None:
            rng = np.random.default_rng(seed=seed)

        hidden_samples = rng.multinomial(n, self.weights)

        return [
            gaussian.sample(n=ncomponent_samples, rng=rng)
            for (ncomponent_samples, gaussian) in zip(hidden_samples, self.gaussians)
        ]

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        """Compute the pdf for the requested samples"""

        weighted_pdf_per_component = [
            self.pdf_contrib(i, samples) for i in range(self.num_components)
        ]
        return np.sum(weighted_pdf_per_component, axis=0)

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        """Compute the log pdf for the requested samples"""

        log_pdf_contribs = np.array(
            [self.log_pdf_contrib(i, samples) for i in range(self.num_components)]
        )
        max_log_pdf_contrib = np.max(log_pdf_contribs, axis=0, keepdims=True)

        log_pdf_contribs -= max_log_pdf_contrib

        return max_log_pdf_contrib[0, :] + np.log(np.exp(log_pdf_contribs).sum(axis=0))

    def pdf_contrib(self, component_idx: int, samples: np.ndarray) -> np.ndarray:
        """Computes pdf contribution of specific component for the requested samples"""

        return self.weights[component_idx] * np.exp(
            self.gaussians[component_idx].log_pdf(samples)
        )

    def log_pdf_contrib(self, component_idx: int, samples: np.ndarray) -> np.ndarray:
        """Computes weighted log pdf of specific component for the requested samples"""

        log_weight = np.log(self.weights[component_idx])
        log_gauss_pdf = self.gaussians[component_idx].log_pdf(samples)

        return log_weight + log_gauss_pdf


class ExpectationMaximisationGMM:
    """Object performing Expectation Maximisation in order to fit a GMM"""

    def __init__(self, num_components: int, seed: int | None = None):
        self.num_components = num_components
        self.weights: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None
        self.gmm: GMMSampler | None = None

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

        if not self.is_initialised:
            self.initialise_params(samples)

        log_likelihood_t0 = -np.inf

        for _ in tqdm(range(max_iter)):
            log_posteriors, log_px = self.perform_e_step(samples)

            log_likelihood_t1 = log_px.sum()

            if log_likelihood_t1 - log_likelihood_t0 < 1e-7:
                break

            log_likelihood_t0 = log_likelihood_t1

            self.perform_m_step(samples, log_posteriors)
            self.update_gmm()

        self.__is_fitted = True

    def compute_upper_bound(
        self, samples: np.ndarray, log_posteriors: np.ndarray
    ) -> float:
        """Compute the EM algorithm upper bounds with the current GMM estimates (t+1)

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension
        posteriors: np.ndarray
            r_{ik} = P(h_i | x_i, theta^(t)), posteriors at previous state.
            Shape (N, K) where N is the number of samples and K the number of components

        Returns
        -------
        float
            sum_{i, k} r_{ik} * log (lambda_k * gaussian_k.pdf(samples_i) / rik)
        """

        log_gaussian_components = np.stack(
            [self.gmm.log_pdf_contrib(i, samples) for i in range(self.num_components)],
            axis=-1,
        )

        upper_bound_terms = log_gaussian_components - log_posteriors
        upper_bound_terms *= np.exp(log_posteriors)

        return upper_bound_terms.sum()

    def perform_e_step(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the E-Step in the expectation maximisation algorithm
        It consists in computing the posterior over the hidden variables


        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        np.ndarray
            log P(h_i | x_i, theta^(t)), posteriors over all samples and components.
            Shape (N, K) where N is the number of samples and K the number of components
        np.ndarray
            log P(x_i) for all samples
        """

        return self.compute_log_posteriors(samples)

    def perform_m_step(self, samples, log_posteriors) -> None:
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

        posteriors = np.exp(log_posteriors)
        unnormalized_weights = posteriors.sum(axis=0) + np.finfo(np.float64).eps  # (K,)

        self.update_weights(unnormalized_weights)
        self.update_means(samples, posteriors, unnormalized_weights)
        self.update_covs(samples, posteriors, unnormalized_weights)

    def compute_posteriors(self, samples: np.ndarray) -> np.ndarray:
        """Compute the posterior P(h_i | x_i)

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        np.ndarray
            Shape (N, K) indicatxing the posterior for each pair h_i
        """

        return np.exp(self.compute_log_posteriors(samples)[0])  # (N, K)

    def compute_log_posteriors(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the log posterior log P(h_i | x_i)

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        log_posterior: np.ndarray
            Shape (N, K) indicatxing the log posterior for each pair h_i
        log_evidence: np.ndarray
            Shape (N,) indicatxing the log P(x) for each sample
        """

        log_pdfs = [
            self.gmm.log_pdf_contrib(i, samples) for i in range(self.num_components)
        ]
        log_pdfs = np.stack(log_pdfs, axis=1)  # (N, K)

        log_pdf_max_per_sample = np.max(log_pdfs, axis=1, keepdims=True)
        normalised_pdfs = np.exp(log_pdfs - log_pdf_max_per_sample)
        log_sum_exp = log_pdf_max_per_sample + np.log(
            np.sum(normalised_pdfs, axis=1, keepdims=True)
        )

        return log_pdfs - log_sum_exp, log_sum_exp[:, 0]

    def update_weights(self, unnormalized_weights: np.ndarray) -> None:
        """Update the GMM weights using the Maximum Likelihood formula"""

        self.weights = unnormalized_weights / unnormalized_weights.sum()

    def update_means(
        self,
        samples: np.ndarray,
        posteriors: np.ndarray,
        unnormalized_weights: np.ndarray,
    ) -> None:
        """Update the GMM means using the Maximum Likelihood formula"""

        weighted_samples = (
            samples[:, np.newaxis, :] * posteriors[..., np.newaxis]
        )  # (N, K, D)
        self.means = weighted_samples.sum(axis=0) / unnormalized_weights[:, np.newaxis]

    def update_covs(
        self,
        samples: np.ndarray,
        posteriors: np.ndarray,
        unnormalized_weights: np.ndarray,
    ) -> None:
        """Update the GMM covariances using the Maximum Likelihood formula"""

        x_mu = samples[:, np.newaxis, :] - self.means[np.newaxis, :, :]  # (N, K, D)

        self.covs = np.zeros((self.num_components, self.ndims, self.ndims))
        for k in range(self.num_components):
            self.covs[k, ...] = (posteriors[:, k] * x_mu[:, k, :].T) @ x_mu[:, k, :]

        self.covs /= unnormalized_weights[:, np.newaxis, np.newaxis]
        self.covs += 1e-6 * np.eye(self.ndims)[np.newaxis, ...]

    def update_gmm(self):
        """Updates the parameters of the estimated gmm"""
        self.gmm.weights = self.weights
        for k in range(self.num_components):
            self.gmm.gaussians[k].mean = self.means[k]
            self.gmm.gaussians[k].cov = self.covs[k]

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """For each sample predicts which component is most likely to have generated it

        Parameters
        ----------
        samples: np.ndarray
            Shape (N, D) where N is the number of samples and D the dimension

        Returns
        -------
        np.ndarray
            Integer array of shape (N,) with taking values in {0, 1, ..., K-1}
            where K is the number Gaussian components
        """
        if not self.is_fitted:
            raise ValueError("GMM has not been fitted, cannot make predictions.")

        posterior = self.compute_posteriors(samples)  # (N, K)

        return np.argmax(posterior, axis=1)

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
        cov_init = np.diag(samples_extent)
        self.covs = np.stack(self.num_components * [cov_init], axis=0)
        self.gmm = GMMSampler(
            weights=self.weights,
            gaussians=[
                Gaussian(mean=mean, cov=cov) for mean, cov in zip(self.means, self.covs)
            ],
        )

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

    @property
    def is_initialised(self) -> bool:
        return self.ndims is not None

    def __str__(self):
        with np.printoptions(precision=3):
            return (
                "GMM with the following parameters \n"
                f"* weights: \n{self.weights} \n\n"
                f"* means  : \n{self.means} \n\n"
                f"* covs   : \n{self.covs}"
            )
