import numpy as np
from scipy.stats import nbinom, truncnorm
from glmpy.models.base_glm import BaseGLM
import torch


class NBModel(BaseGLM):
    def __init__(self, y, **kwargs):
        super().__init__(y=y, **kwargs)
        self.has_mle = True
        
    def get_initial_params(self):
        epsilon = 1e-10
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()
        num_betas = X_numpy.shape[1]

        scaling_factor = 0.1
        theoretical_mus = np.array(
            [
                np.log(np.mean(y_numpy[X_numpy[:, i] != 0]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        # theoretical_mus = np.nan_to_num(theoretical_mus, nan=epsilon)


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
        # theoretical_alphas = np.nan_to_num(theoretical_alphas, nan=epsilon)

        initial_mus = [
            [np.random.normal(loc=theoretical_mus[i], scale=abs(scaling_factor * theoretical_mus[i])) for i in range(num_betas)]
            for _ in range(5)
        ] + [theoretical_mus.tolist()]

        initial_alphas = [
            [np.random.normal(loc=theoretical_alphas[i], scale=abs(scaling_factor * theoretical_alphas[i])) for i in range(num_betas)]
            for _ in range(5)
        ] + [theoretical_alphas.tolist()]

        initial_params = [
            initial_mus[i] + initial_alphas[i]
            for i in range(6)
        ]

        return initial_params

    def loglik(self, params, params_tensor=None, ret_torch=True):
        if params_tensor is not None:
            params = params_tensor
            
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params).reshape(-1, 1)

        num_betas = self.X.shape[1]
        beta = params[:num_betas]
        alpha = params[num_betas:]  # Dispersion parameter

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

        ll = torch.distributions.NegativeBinomial(total_count=r, probs=p).log_prob(self.y).sum()

        if ret_torch:
            return self.scale_factor * -ll
        return self.scale_factor * -ll.detach()
    
        
    @staticmethod
    def sample(params, size=10000, design_matrix=None):
        num_betas = len(params) // 2
        
        betas = params[:num_betas]
        alphas = params[num_betas:]
        
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
        
        samples = nbinom.rvs(r, p, size=size)
            
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
        
        num_betas = len(params) // 2

        epsilon = 1e-8
        
        r = params[num_betas + covariate_index]
        r = np.nan_to_num(r, nan = epsilon)
        r = np.clip(r, epsilon, 1e6)
        p = r / (r + np.exp(params[covariate_index]))
        p = np.nan_to_num(p, nan = epsilon)
        p = np.clip(p, epsilon, 1 - epsilon)

        samples = nbinom.rvs(r, p, size=size)           
            
        return samples
    
    @staticmethod
    def pmf_range(params, upper_bound, lower_bound=0, design_matrix=None, mu=None, alpha=None):
        size = upper_bound - lower_bound + 1
        if size <= 0:
            raise ValueError("upper_bound must be greater than lower_bound.")
        
        betas = params[:len(params)//2]
        alphas = params[len(params)//2:]
        
        if mu is None:
            mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=design_matrix)
            
        if alpha is None:
            alpha = BaseGLM._get_linear_estimator(alphas, size, num_covariates=len(alphas), design_matrix=design_matrix)

        # Calculate parameters for the Negative Binomial distribution
        r = alpha
        r = np.clip(r, 1e-7, 1 - 1e-7)

        p = r / (r + mu)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        x = np.arange(lower_bound, upper_bound + 1)
        pmf = nbinom.pmf(x, r, p)

        return pmf