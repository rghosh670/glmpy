from pprint import pprint

import numpy as np
import torch
from scipy.special import expit, logit
from scipy.stats import poisson

from glmpy.core.models.base_glm import BaseGLM


class ZIPModel(BaseGLM):
    def __init__(self, y, **kwargs):
        super().__init__(y=y, **kwargs)
        self.has_mle = False
        

    def get_initial_params(self):
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()

        num_betas = X_numpy.shape[1]

        theoretical_mus = np.array(
            [
                np.log(np.mean(y_numpy[(X_numpy[:, i] != 0) & (y_numpy != 0)]) + 1e-4)
                for i in range(num_betas)
            ]
        )

        theoretical_pi = len(y_numpy[y_numpy == 0]) / len(y_numpy)
        theoretical_pi = max(min(theoretical_pi, 0.999), 0.001)
        theoretical_p0 = logit(theoretical_pi)

        # Define the scaling factor for sigma
        scaling_factor = 0.1  # for example, 1% of the mean

        # Generate initial params with scaled sigma
        initial_params = [
            [
                np.random.normal(
                    loc=theoretical_mus[i],
                    scale=abs(scaling_factor * theoretical_mus[i]),
                )
                for i in range(num_betas)
            ]
            + [
                np.random.normal(
                    loc=theoretical_p0, scale=abs(scaling_factor * theoretical_p0)
                )
            ]
            for _ in range(5)
        ] + [theoretical_mus.tolist() + [theoretical_p0]]

        return initial_params

    def loglik(self, params, params_tensor=None, ret_torch=True):
        if params_tensor is not None:
            params = params_tensor

        beta = params[:-1]
        p0 = torch.sigmoid(params[-1])
        mu = torch.matmul(self.X, torch.exp(beta))

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
        covariate_list = ['!Intercept'] + sorted(covariate_list)
        covariate_index = covariate_list.index(covariate)
        
        dm = design_matrix[:, [0, covariate_index]]
        
        betas = [params[0], params[covariate_index]]
        p0 = params[-1]
        
        mu = np.dot(dm, np.exp(betas))
        
        pi = expit(p0)

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