import os

import numpy as np
import pymc as pm
import pytensor
import torch
from icecream import ic
from scipy.special import expit, logit
from scipy.stats import nbinom, truncnorm

from glmpy.models.base_glm import BaseGLM

class ZINBModel(BaseGLM):
    def __init__(self, y, **kwargs):
        super().__init__(y=y, **kwargs)
        self.has_mle = False

    def get_initial_params(self):
        epsilon = 1e-4
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()

        num_betas = X_numpy.shape[1]
        
        theoretical_pi = len(y_numpy[y_numpy == 0]) / len(y_numpy)
        theoretical_pi = max(min(theoretical_pi, 0.99), 0.01)
        theoretical_p0 = logit(theoretical_pi)

        scaling_factor = 0.1
        num_initial_samples = 6
        
        # ic(theoretical_p0)

        # First set of theoretical mus
        theoretical_mus = np.array(
            [
                np.log(np.mean(y_numpy[X_numpy[:, i] != 0]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        # ic(theoretical_mus)

        # Second set of theoretical mus excluding zeros
        theoretical_mus_nonzero = np.array(
            [
                np.log(np.mean(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        
        # ic(theoretical_mus_nonzero)

        # Determine the length of the linspace arrays
        max_len = num_initial_samples * 2 - 2  # Since [1:-1] removes the first and last elements

        # Generate evenly spaced values between mus and mus_nonzero
        theoretical_mus_spaced = np.array(
            [
                np.linspace(theoretical_mus[i], theoretical_mus_nonzero[i], num_initial_samples * 2)[1:-1]
                if np.any(X_numpy[:, i] != 0) else np.full(max_len, epsilon)
                for i in range(num_betas)
            ]
        ).T
        
        # ic(theoretical_mus_spaced)

        # Theoretical alphas based on theoretical_mus
        theoretical_alphas = np.array(
            [
                np.exp(theoretical_mus[i]) ** 2 / (np.var(y_numpy[X_numpy[:, i] != 0]) - np.exp(theoretical_mus[i]))
                if np.any(X_numpy[:, i] != 0)
                else 1e-6
                if np.var(y_numpy[X_numpy[:, i] != 0]) <= np.exp(theoretical_mus[i])
                else 1e6  # Large value for non-overdispersed data
                for i in range(num_betas)
            ]
        )
        
        # ic(theoretical_alphas)

        # Theoretical alphas based on theoretical_mus_nonzero
        theoretical_alphas_nonzero = np.array(
            [
                (np.exp(theoretical_mus_nonzero[i]) ** 2) / (np.var(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) - np.exp(theoretical_mus_nonzero[i]))
                if np.any((X_numpy[:, i] != 0) & (y_numpy != 0)) and np.var(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) > np.exp(theoretical_mus_nonzero[i])
                else epsilon
                if not np.any((X_numpy[:, i] != 0))
                else 1e6
                if np.var(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) <= np.exp(theoretical_mus_nonzero[i])
                else epsilon
                for i in range(num_betas)
            ]
        )

        # ic(theoretical_alphas_nonzero)
        
        # Theoretical alphas based on theoretical_mus_spaced
        theoretical_alphas_spaced = np.array(
            [
                [
                    (np.exp(spaced_mu) ** 2) / (np.var(y_numpy[X_numpy[:, i] != 0]) - np.exp(spaced_mu))
                    if np.any(X_numpy[:, i] != 0) and np.var(y_numpy[X_numpy[:, i] != 0]) > np.exp(spaced_mu)
                    else epsilon
                    if not np.any(X_numpy[:, i] != 0)
                    else 1e6
                    if np.var(y_numpy[X_numpy[:, i] != 0]) <= np.exp(spaced_mu)
                    else epsilon
                    for spaced_mu in theoretical_mus_spaced[:, i]
                ]
                for i in range(num_betas)
            ]
        ).T
        # ic(theoretical_alphas_spaced)

        initial_mus_nonzero = [
            [
                np.random.normal(
                    loc=theoretical_mus_nonzero[i],
                    scale=abs(scaling_factor * theoretical_mus_nonzero[i]),
                )
                for i in range(num_betas)
            ]
            for _ in range(num_initial_samples)
        ] + [theoretical_mus_nonzero.tolist()]



        initial_alphas_nonzero = [
            [
                np.random.normal(
                    loc=theoretical_alphas_nonzero[i],
                    scale=abs(scaling_factor * theoretical_alphas_nonzero[i]),
                )
                for i in range(num_betas)
            ]
            for _ in range(num_initial_samples)
        ] + [theoretical_alphas_nonzero.tolist()]


        initial_p0s = [
            [
                np.random.normal(
                    loc=theoretical_p0, scale=abs(scaling_factor * theoretical_p0)
                )
            ]
            for _ in range(num_initial_samples * 3 - 1)
        ] + [[theoretical_p0]]

        initial_params = [
            np.hstack((np.nan_to_num(initial_mus_nonzero[i], nan = -6), initial_p0s[i], initial_alphas_nonzero[i])) for i in range(num_initial_samples)
        ]

        # ic(initial_params)

        return initial_params


    def loglik(self, params, params_tensor=None, ret_torch=True):
        if params_tensor is not None:
            params = params_tensor
            
        # ic(params)
            
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params).reshape(-1, 1)

        num_betas = self.X.shape[1]
        beta = params[:num_betas]

        p0 = params[num_betas]

        pi = torch.sigmoid(params[num_betas])
        alpha = params[num_betas + 1:]
        
        mu = torch.matmul(self.X[:, 1:], torch.exp(beta[1:]))
        sigma = torch.matmul(self.X[:, 1:], alpha[1:])

        r = sigma
        epsilon = 1e-6
        r = torch.nan_to_num(r, nan = epsilon)
        r = torch.clamp(r, min=epsilon, max=1e5)
        r[r < epsilon] = epsilon

        p = sigma / (sigma + mu)

        p = torch.nan_to_num(p, nan = epsilon)
        p = torch.clamp(p, epsilon, 1 - epsilon)
    
        pmf_zero = torch.distributions.NegativeBinomial(
            total_count=r, probs=p
        ).log_prob(torch.zeros_like(self.y))
        
        
        pmf_y = torch.distributions.NegativeBinomial(total_count=r, probs=p).log_prob(
            self.y
        )

        ll = torch.where(
            self.y == 0,
            torch.logaddexp(torch.log(pi), torch.log1p(-pi) + pmf_zero),
            torch.log1p(-pi) + pmf_y
        ).sum()       

        if ret_torch:
            return self.scale_factor * -ll
        return self.scale_factor * -ll.detach()

    @staticmethod
    def sample(params, design_matrix=None, size=10000):
        """
        Sample from a Zero-Inflated Negative Binomial (ZINB) distribution.

        Parameters:
        - params: array-like, parameters of the model.
                params[0] to params[-3] are beta coefficients (including intercept if present)
                params[-2] is the logit-transformed zero-inflation probability (p0).
                params[-1] is the dispersion parameter (alpha).
        - design_matrix: array-like, optional, the design matrix used for prediction.
        - size: int, number of samples to generate.

        Returns:
        - samples: array of samples.
        """
        # Extract intercept, betas, zero-inflation logit parameter, and dispersion
        num_betas = len(params) // 2
        
        betas = params[:num_betas]
        p0 = params[num_betas]
        alphas = params[num_betas + 1:]

        mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=design_matrix)
        sigma = BaseGLM._get_linear_estimator(alphas, size, num_covariates=len(alphas), design_matrix=design_matrix)

        r = sigma
        epsilon = 1e-8
        r = np.nan_to_num(r, nan = epsilon)
        r = np.clip(r, epsilon, 1e6)
        r[r < epsilon] = epsilon

        p = sigma / (sigma + mu)

        p = np.nan_to_num(p, nan = epsilon)
        p = np.clip(p, epsilon, 1 - epsilon)

        # Calculate zero-inflation probability
        pi = expit(p0)        
        
        nb_samples = nbinom.rvs(r, p, size=size)
        zero_inflated = np.random.binomial(1, pi, size=size)

        # Samples: if zero_inflated is 1, set sample to 0, otherwise keep NB sample
        samples = np.where(zero_inflated == 1, 0, nb_samples)

        return samples
    
    @staticmethod
    def sample_from_covariate(params, covariate, covariate_list, size=10000, design_matrix=None):
        """
        Sample from a Poisson distribution.

        Parameters:
        - params: array-like, parameters of the model.
        - size: int, number of samples to generate.
        - design_matrix: array-like, optional, design matrix for the covariates.

        Returns:
        - samples: array of Poisson-distributed samples.
        """
        covariate_list = sorted(covariate_list)
        covariate_index = covariate_list.index(covariate) + 1
        
        # dm = design_matrix[:, [0, covariate_index]]
        # dm = dm[dm[:, 1] != 0]
        
        num_betas = len(params) // 2
        
        # betas = [0, params[covariate_index]]
        p0 = params[num_betas]
        # alphas = [params[0], params[num_betas + covariate_index + 1]]
        
        # mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=dm)
        # sigma = BaseGLM._get_linear_estimator(alphas, size, num_covariates=len(alphas), design_matrix=dm)
        
        # r = sigma
        # p = r / (r + mu)
        
        # ic(r)
        # ic(p)
        
        pi = expit(p0)
        
        epsilon = 1e-8
        
        r = params[num_betas + covariate_index]
        r = np.nan_to_num(r, nan = epsilon)
        r = np.clip(r, epsilon, 1e6)
        p = r / (r + np.exp(params[covariate_index]))
        p = np.nan_to_num(p, nan = epsilon)
        p = np.clip(p, epsilon, 1 - epsilon)

        
        nb_samples = nbinom.rvs(r, p, size=size)
        zero_inflated = np.random.binomial(1, pi, size=size)
        
        samples = np.where(zero_inflated == 1, 0, nb_samples)
        
        return samples
    
    @staticmethod
    def pmf_range(params, upper_bound, lower_bound=0, design_matrix=None, mu=None, alpha=None):
        size = upper_bound - lower_bound + 1
        if size <= 0:
            raise ValueError("upper_bound must be greater than lower_bound.")
        
        num_betas = len(params) // 2
        
        betas = params[:num_betas]
        p0 = params[num_betas]
        alphas = params[num_betas + 1:]
        
        if mu is None:
            mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=design_matrix)
            
        if alpha is None:
            alpha = BaseGLM._get_linear_estimator(alphas, size, num_covariates=len(alphas), design_matrix=design_matrix)

        # Calculate parameters for the Negative Binomial distribution
        r = alpha
        r = np.clip(r, 1e-7, 1 - 1e-7)

        p = r / (r + mu)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        
        p0 = params[-1] if p0 is None else p0
        pi = expit(p0)

        x = np.arange(lower_bound, upper_bound + 1)
        
        nb_pmf = nbinom.pmf(x, r, p)
        zinb_pmf = (1 - pi) * nb_pmf
        zinb_pmf[0] = pi
        
        return zinb_pmf
    
    def empirical_bayes_inference(self, fixed_sigma=1.0, num_samples=250, tune=250):
        """
        Perform empirical Bayes inference using best_params without relying on the covariance matrix.

        Parameters:
        - fixed_sigma: float, standard deviation for priors (set to control prior uncertainty).
        - num_samples: int, number of samples for MCMC.
        - tune: int, number of tuning steps for MCMC.

        Returns:
        - trace: MCMC trace containing sampled posterior distributions.
        """
        if self.best_params is None:
            raise ValueError(
                "GLM fit not found. Please run the GLM fitting first to obtain initial estimates."
            )

        with pm.Model() as model:
            # Use best_params as the mean for priors and set a fixed standard deviation
            beta = pm.Normal(
                "beta",
                mu=self.best_params[:-2],
                sigma=fixed_sigma,
                shape=len(self.best_params[:-2]),
            )

            # Priors for p0 and theta, using best_params and a fixed sigma
            p0 = pm.Normal("p0", mu=self.best_params[-2], sigma=fixed_sigma)
            theta = pm.HalfNormal("theta", sigma=fixed_sigma)

            mu = pm.math.exp(pm.math.dot(self.X, beta))
            mu = pm.math.dot(self.X, pm.math.exp(beta))
            p = pm.math.invlogit(p0)

            y_obs = pm.ZeroInflatedNegativeBinomial(
                "y_obs", psi=p, mu=mu, alpha=theta, observed=self.y
            )

            trace = pm.sample(
                num_samples, tune=tune, chains=2, cores=2, progressbar=False
            )

        self.bayes_idata = trace
        return trace
