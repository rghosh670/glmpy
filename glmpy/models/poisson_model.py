import numpy as np
from scipy.stats import poisson
from glmpy.models.base_glm import BaseGLM
import torch

class PoissonModel(BaseGLM):
    def __init__(self, y, **kwargs):
        super().__init__(y=y, **kwargs)
        self.has_mle = True

    def get_initial_params(self):
        epsilon = 1e-6
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()

        num_betas = X_numpy.shape[1]

        theoretical_mus = np.array(
            [
                np.log(np.mean(y_numpy[X_numpy[:, i] != 0]) + epsilon)
                if np.any(X_numpy[:, i] != 0) else np.log(epsilon)
                for i in range(num_betas)
            ]
        )
        
        # theoretical_mus = np.nan_to_num(theoretical_mus, nan=np.log(epsilon))

        # Define the scaling factor for sigma
        scaling_factor = 0.1  # for example, 1% of the mean

        # Generate initial params with scaled sigma
        initial_params = [
            [np.random.normal(loc=theoretical_mus[i], scale=abs(scaling_factor * theoretical_mus[i])) for i in range(num_betas)]
            for _ in range(5)
        ] + [theoretical_mus.tolist()]

        return initial_params

    def loglik(self, params, params_tensor = None, ret_torch=True):
        params = params if params_tensor is None else params_tensor
        
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params).reshape(-1, 1)
            
        beta = params
        mu = torch.matmul(self.X[:, 1:], torch.exp(beta[1:]))

        # Calculate the log-likelihood
        ll = torch.distributions.Poisson(mu).log_prob(self.y).sum()

        if ret_torch:
            return self.scale_factor * -ll

        return self.scale_factor * -ll.detach()
    
    def estimate_with_mle(self):
        y_numpy = self.y.detach().numpy()
        X_numpy = self.X.detach().numpy()
        num_betas = X_numpy.shape[1]
        
        return np.array([
            np.log(np.mean(y_numpy[X_numpy[:, i] != 0]) + 1e-4) if np.any(X_numpy[:, i] != 0) else 1e-6
            for i in range(num_betas)
        ])
        

    @staticmethod
    def sample(params, size=10000, design_matrix=None):
        """
        Sample from a Poisson distribution.

        Parameters:
        - params: array-like, parameters of the model.
        - size: int, number of samples to generate.
        - design_matrix: array-like, optional, design matrix for the covariates.

        Returns:
        - samples: array of Poisson-distributed samples.
        """
        mu = BaseGLM._get_linear_estimator(params, size, num_covariates=len(params), design_matrix=design_matrix)
        samples = np.random.poisson(mu, size=size)
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
        mu = BaseGLM._get_linear_estimator(betas, size, num_covariates=len(betas), design_matrix=dm)
        samples = np.random.poisson(mu, size=size)
        return samples

    @staticmethod
    def pmf_range(params, upper_bound, lower_bound=0, design_matrix=None, mu=None):
        """
        Compute the Poisson PMF from a lower bound to an upper bound.

        Parameters:
        - params: array-like, parameters of the model.
        - upper_bound: int, the upper bound of the range.
        - lower_bound: int, the lower bound of the range.
        - design_matrix: array-like, optional, design matrix for the covariates.

        Returns:
        - pmf_values: array, PMF values from lower_bound to upper_bound.
        """
        size = upper_bound - lower_bound + 1
        if size <= 0:
            raise ValueError("upper_bound must be greater than lower_bound.")

        if mu is None:
            mu = BaseGLM._get_linear_estimator(params, size, num_covariates=len(params), design_matrix=design_matrix)
        x = np.arange(lower_bound, upper_bound + 1)
        pmf = poisson.pmf(x, mu)
        return pmf