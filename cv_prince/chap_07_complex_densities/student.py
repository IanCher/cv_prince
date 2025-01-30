"""Contains scripts related to the estimation of student t distributions"""

import warnings
import numpy as np
import scipy.optimize
import scipy.special
from tqdm import tqdm


class ExpectationMaximisationStudent:
    """Object performing Expectation Maximisation to fit a student t distribution"""

    def __init__(self, seed: int | None = None):
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.nu: float | None = None

        self.ndims: int | None = None

        self.__rng = np.random.default_rng(seed)
        self.__is_fitted: bool = False

    def fit(self, samples: np.ndarray, max_iter: int = 1000) -> None:
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
        nsamples = samples.shape[0]

        log_px = scipy.special.loggamma((self.nu + self.ndims) / 2)
        log_px -= scipy.special.loggamma(self.nu / 2)
        log_px -= self.ndims / 2 * np.log(self.nu * np.pi)
        log_px -= 1 / 2 * np.log(np.linalg.det(self.cov))
        log_px *= nsamples

        x_mu_sq = self.compute_scaled_dist_to_mean(samples)
        log_px += (self.nu + self.ndims) / 2 * np.log(1 + x_mu_sq / self.nu).sum()

        return log_px

    def initialise_params(self, samples: np.ndarray) -> None:
        self.ndims = samples.shape[1]

        self.mean = np.mean(samples, axis=0)
        self.cov = np.cov(samples, rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov)
        self.nu = 1000

    def perform_e_step(self, samples) -> tuple[np.ndarray, np.ndarray]:
        alpha = (self.nu + self.ndims) / 2

        x_mu_sq = self.compute_scaled_dist_to_mean(samples)
        beta = (x_mu_sq + self.nu) / 2

        exp_h = alpha / beta
        exp_logh = scipy.special.digamma(alpha) - np.log(beta)

        return exp_h, exp_logh

    def perform_m_step(
        self, samples: np.ndarray, exp_h: np.ndarray, exp_logh: np.ndarray
    ) -> None:
        nsamples = samples.shape[0]

        self.mean = np.sum(exp_h[:, np.newaxis] * samples, axis=0) / exp_h.sum()

        x_mu = samples - self.mean[np.newaxis, :]  # (N, D)
        x_mu_sq = x_mu[..., np.newaxis] @ x_mu[:, np.newaxis, :]  # (N, D, D)
        self.cov = (
            1 / nsamples * np.sum(exp_h[:, np.newaxis, np.newaxis] * x_mu_sq, axis=0)
        )
        self.inv_cov = np.linalg.inv(self.cov)

        def cost_func_nu(nu):
            nu_2 = nu / 2
            term1 = nsamples * nu_2 * np.log(nu_2)
            term2 = -nsamples * scipy.special.loggamma(nu_2)
            term3 = (nu_2 - 1) * exp_logh.sum()
            term4 = -nu_2 * exp_h.sum()

            return -(term1 + term2 + term3 + term4)

        def cost_func_nu_grad(nu):
            nu_2 = nu / 2
            avg_diff_exp_logh_h = np.sum(exp_logh - exp_h)
            avg_diff_exp_logh_h /= nsamples
            digamma_nu_2 = scipy.special.digamma(nu_2)
            log_nu_2 = np.log(nu_2)
            cost = log_nu_2 + 1 - digamma_nu_2 + avg_diff_exp_logh_h

            return -(nsamples / 2 * cost)

        nu_init = self.nu
        dir_init = -np.sign(cost_func_nu_grad(nu_init)) * 0.99 * np.abs(nu_init)

        res = scipy.optimize.line_search(
            cost_func_nu,
            cost_func_nu_grad,
            xk=nu_init,
            pk=dir_init,
        )

        if res[0] is not None:
            self.nu = nu_init + res[0] * dir_init
        else:
            warnings.warn(
                "Line search did not converge, keeping the same nu parameter."
            )

    def compute_scaled_dist_to_mean(self, samples: np.ndarray) -> np.ndarray:
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
                f"* nu  : \n{self.nu} \n"
            )
