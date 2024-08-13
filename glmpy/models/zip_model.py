from pprint import pprint

import numpy as np
import torch
from scipy.special import expit, logit
from scipy.stats import poisson

from glmpy.models.base_glm import BaseGLM

class ZIPModel(BaseGLM):
    def __init__(self, y, **kwargs):
        super().__init__(y=y, **kwargs)
        self.has_mle = False
        
    def get_initial_params(self):
        epsilon = 1e-6
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()
        num_initial_samples = 5

        num_betas = X_numpy.shape[1]

        # First set of theoretical mus for non-zero y and non-zero X
        theoretical_mus = np.array(
            [
                np.log(np.mean(y_numpy[X_numpy[:, i] != 0]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        # theoretical_mus = np.nan_to_num(theoretical_mus, nan=epsilon)

        # Second set of theoretical mus including zeros in y
        theoretical_mus_nonzero = np.array(
            [
                np.log(np.mean(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        # theoretical_mus_nonzero = np.nan_to_num(theoretical_mus_nonzero, nan=np.log(epsilon))

        # Generate evenly spaced values between mus and mus_zero_included
        # Determine the length of the linspace arrays
        max_len = num_initial_samples * 2 - 2  # Since [1:-1] removes the first and last elements

        # Generate evenly spaced values between mus and mus_nonzero
        theoretical_mus_spaced = np.array(
            [
                np.linspace(-13.8, theoretical_mus_nonzero[i], num_initial_samples * 2)[1:-1]
                if np.any(X_numpy[:, i] != 0) else np.full(max_len, epsilon)
                for i in range(num_betas)
            ]
        ).T

        # theoretical_mus_spaced = np.nan_to_num(theoretical_mus_spaced, nan=np.log(epsilon))

        theoretical_pi = len(y_numpy[y_numpy == 0]) / len(y_numpy)
        theoretical_pi = max(min(theoretical_pi, 0.999), 0.001)
        theoretical_p0 = logit(theoretical_pi)

        scaling_factor = 0.1

        initial_mus = [
            [
                np.random.normal(
                    loc=theoretical_mus[i],
                    scale=abs(scaling_factor * theoretical_mus[i]),
                )
                for i in range(num_betas)
            ]
            for _ in range(num_initial_samples)
        ] + [theoretical_mus.tolist()]

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

        initial_mus_spaced = [
            spaced.tolist()
            for spaced in theoretical_mus_spaced
        ]

        initial_p0s = [
            [
                np.random.normal(
                    loc=theoretical_p0, scale=abs(scaling_factor * theoretical_p0)
                )
            ]
            for _ in range(num_initial_samples * 3 - 1)
        ] + [[theoretical_p0]]

        initial_params = [
            np.hstack((np.nan_to_num(initial_mus[i], np.log(epsilon)), initial_p0s[i]))
            for i in range(num_initial_samples)
        ]

        initial_params += [
            np.hstack((np.nan_to_num(initial_mus_nonzero[i - num_initial_samples], np.log(epsilon)), initial_p0s[i]))
            for i in range(num_initial_samples, num_initial_samples * 2)
        ]

        initial_params += [
            np.hstack((np.nan_to_num(initial_mus_spaced[i - num_initial_samples * 2], np.log(epsilon)), initial_p0s[i]))
            for i in range(num_initial_samples * 2, num_initial_samples * 3)
        ]

        return initial_params

    def loglik(self, params, params_tensor=None, ret_torch=True):
        if params_tensor is not None:
            params = params_tensor
            
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params).reshape(-1, 1)

        beta = params[:-1]
        p0 = torch.sigmoid(params[-1])
        mu = torch.matmul(self.X[:, 1:], torch.exp(beta[1:]))

        # Calculate pmf for zero and non-zero counts
        pmf_zero = torch.exp(-mu)
        pmf_y = torch.exp(
            -mu + self.y * torch.log(mu) - torch.lgamma(self.y + 1)
        )

        # Ensure numerical stability
        pmf_zero = torch.clamp(pmf_zero, min=1e-10)
        pmf_y = torch.clamp(pmf_y, min=1e-10)

        # Calculate log-likelihood
        ll = torch.where(
            self.y == 0,
            torch.log(p0 + (1 - p0) * pmf_zero),
            torch.log((1 - p0) * pmf_y),
        ).sum()

        if ret_torch:
            return self.scale_factor * -ll
        return self.scale_factor * -ll.detach()

    @staticmethod
    def sample(params, size=10000, design_matrix=None):
        """
        Sample from a Zero-Inflated Poisson (ZIP) distribution.

        Parameters:
        - params: array-like, parameters of the model.
                  params[0] to params[-2] are beta coefficients (including intercept if present)
                  params[-1] is the logit-transformed zero-inflation probability (p0).
        - size: int, number of samples to generate.

        Returns:
        - sampled_data: array of samples.
        """
        # Extract intercept, betas, and zero-inflation logit parameter
        betas = params[:-1]
        p0 = params[-1]

        mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=design_matrix)
        
        pi = expit(p0)

        # Draw samples from the ZIP distribution
        poisson_samples = np.random.poisson(mu, size=size)
        zero_inflated = np.random.binomial(1, pi, size=size)

        # Samples: if zero_inflated is 1, set sample to 0, otherwise keep Poisson sample
        samples = np.where(zero_inflated == 1, 0, poisson_samples)

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
        
        
        dm = design_matrix[:, [0, covariate_index]]
        dm = dm[dm[:, 1] != 0]
        
        betas = [0, params[covariate_index]]
        p0 = params[-1]
        
        pi = expit(p0)
        mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=dm)

        samples = np.random.poisson(mu, size=size)
        zero_inflated = np.random.binomial(1, pi, size=size)
        
        samples = np.where(zero_inflated == 1, 0, samples)
        
        return samples
    
    @staticmethod
    def pmf_range(params, upper_bound, lower_bound=0, design_matrix=None, mu=None, p0=None):
        size = upper_bound - lower_bound + 1
        if size <= 0:
            raise ValueError("upper_bound must be greater than lower_bound.")
        
        betas = params[:-1]
        p0 = params[-1]
        
        if mu is None:
            mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=design_matrix)
            
        p0 = params[-1] if p0 is None else p0
        pi = expit(p0)
    
        x = np.arange(lower_bound, upper_bound + 1)
        
        # Calculate the PMF for Zero-Inflated Poisson
        poisson_pmf = poisson.pmf(x, mu)
        zip_pmf = (1 - pi) * poisson_pmf
        zip_pmf[0] += pi  # Adding the inflation part for zero
        
        return zip_pmf