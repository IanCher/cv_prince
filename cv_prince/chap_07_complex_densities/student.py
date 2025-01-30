"""Contains scripts related to the estimation of student t distributions"""

import warnings
import numpy as np
import scipy.optimize
import scipy.special
from tqdm import tqdm


class ExpectationMaximisationStudent:
    """Object performing Expectation Maximisation to fit a student t distribution

    Parameters
    ----------
    mean: np.ndarray | None
        Current estimate of the mean, None if EM has not been initialised
    cov: np.ndarray | None
        Current estimate of the covariance matrix, None if EM has not been initialised
    inv_cov: np.ndarray | None
        Store the inverse of cov to avoid recomputing all the time
    df: float | None
        Degrees of freedom, always positive, None if EM has not been intialised
    dim: int | None
        Data dimension, None if EM has not been initialised
    """

    def __init__(self, seed: int | None = None):
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.df: float | None = None

        self.dim: int | None = None

        self.__rng = np.random.default_rng(seed)  # RNG, can be initialised with seed
        self.__is_fitted: bool = False  # Indicates whether EM has been fit

    def fit(self, samples: np.ndarray, max_iter: int = 1000) -> None:
        """Fit a student t distribution on data using Expectation Maximisation

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        max_iter: int (default=1000)
            Max number of iterations if no convergence is reached
        """

        self.initialise_params(samples)

        obj_t0 = self.eval_objective(samples)

        for _ in tqdm(range(max_iter)):
            exp_h, exp_logh = self.perform_e_step(samples)
            self.perform_m_step(samples, exp_h, exp_logh)

            obj_t1 = self.eval_objective(samples)
            if obj_t1 - obj_t0 < 1e-7:
                break

            obj_t0 = obj_t1

    def eval_objective(self, samples: np.ndarray) -> np.ndarray:
        """Compute the object using the current paramters estimate
        The objective is sum_i log P(x_i)

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        """
        # pylint: disable=no-member

        nsamples = samples.shape[0]
        log_px = scipy.special.loggamma((self.df + self.dim) / 2)
        log_px -= scipy.special.loggamma(self.df / 2)
        log_px -= self.dim / 2 * np.log(self.df * np.pi)
        log_px -= 1 / 2 * np.log(np.linalg.det(self.cov))
        log_px *= nsamples

        x_mu_sq = self.compute_scaled_dist_to_mean(samples)
        log_px += (self.df + self.dim) / 2 * np.log(1 + x_mu_sq / self.df).sum()

        return log_px

    def initialise_params(self, samples: np.ndarray) -> None:
        """Initialise all the parameters from data

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        """

        self.dim = samples.shape[1]
        self.mean = np.mean(samples, axis=0)
        self.cov = np.cov(samples, rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov)
        self.df = 1000

    def perform_e_step(self, samples) -> tuple[np.ndarray, np.ndarray]:
        """Performs the E-Step consisting in estimating the posterior P(h_i|x_i)

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (Shape (N,), Shape (N,))
            (E[h_i], E[logh_i]) for all samples, with h_i following P(h_i|x_i)
        """

        alpha = (self.df + self.dim) / 2

        x_mu_sq = self.compute_scaled_dist_to_mean(samples)
        beta = (x_mu_sq + self.df) / 2

        exp_h = alpha / beta
        exp_logh = scipy.special.digamma(alpha) - np.log(beta)

        return exp_h, exp_logh

    def perform_m_step(
        self, samples: np.ndarray, exp_h: np.ndarray, exp_logh: np.ndarray
    ) -> None:
        """Performs the M-Step consisting in maximising a lower bound wrt the parameters

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        exp_h: np.ndarray
            Shape (N,), E[h_i] for all samples with h_i following P(h_i|x_i)
        exp_logh: np.ndarray
            Shape (N,), E[logh_i] for all samples with h_i following P(h_i|x_i)
        """
        # pylint: disable=no-member

        nsamples = samples.shape[0]

        self.mean = np.sum(exp_h[:, np.newaxis] * samples, axis=0) / exp_h.sum()

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)
        x_mu_sq = x_mu[..., np.newaxis] @ x_mu[:, np.newaxis, :]  # (N, D, D)
        self.cov = (
            1 / nsamples * np.sum(exp_h[:, np.newaxis, np.newaxis] * x_mu_sq, axis=0)
        )
        self.inv_cov = np.linalg.inv(self.cov)

        def cost_func_df(df):
            df_2 = df / 2
            term1 = nsamples * df_2 * np.log(df_2)
            term2 = -nsamples * scipy.special.loggamma(df_2)
            term3 = (df_2 - 1) * exp_logh.sum()
            term4 = -df_2 * exp_h.sum()

            return -(term1 + term2 + term3 + term4)

        def cost_func_df_grad(df):
            df_2 = df / 2
            avg_diff_exp_logh_h = np.sum(exp_logh - exp_h)
            avg_diff_exp_logh_h /= nsamples
            digamma_df_2 = scipy.special.digamma(df_2)
            log_df_2 = np.log(df_2)
            cost = log_df_2 + 1 - digamma_df_2 + avg_diff_exp_logh_h

            return -(nsamples / 2 * cost)

        df_init = self.df
        dir_init = -np.sign(cost_func_df_grad(df_init)) * 0.99 * np.abs(df_init)

        res = scipy.optimize.line_search(
            cost_func_df,
            cost_func_df_grad,
            xk=df_init,
            pk=dir_init,
        )

        if res[0] is not None:
            self.df = df_init + res[0] * dir_init
        else:
            warnings.warn(
                "Line search did not converge, keeping the same nu parameter."
            )

    def compute_scaled_dist_to_mean(self, samples: np.ndarray) -> np.ndarray:
        """Computes (x - mean)^T * inv_cov * (x - mean)

        Parameters
        ----------
        samples: np.ndarray
            Input samples of shape (N, D) with N the number of sample, D the dimension
        """

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)
        x_mu_sq = x_mu[:, np.newaxis, :] @ self.inv_cov[np.newaxis, ...]  # (N, 1, D)
        x_mu_sq = x_mu_sq @ x_mu[..., np.newaxis]  # (N, 1, 1)

        return x_mu_sq[..., 0, 0]  # (N,)

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
                f"* mean: \n{self.mean} \n\n"
                f"* cov : \n{self.cov} \n\n"
                f"* nu  : \n{self.df} \n"
            )
