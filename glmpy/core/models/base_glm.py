import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import chi2


class BaseGLM(BaseModel):
    """
    A base model class for statistical modeling using PyTorch and SciPy.

    Attributes:
        y (Union[np.ndarray, list, pd.Series]): Target variable.
        X (Optional[np.ndarray]): Feature matrix.
        name (Optional[str]): Model name.
        covariates (Optional[dict]): Dictionary of covariates.
        covariate_names (list): List of covariate names.
        best_fit_res (Optional[OptimizeResult]): Result of the best fit.
        best_params (Optional[np.ndarray]): Best fit parameters.
        best_loglik (float): Best log-likelihood value.
        best_initialization (Optional[np.ndarray]): Initialization that led to the best fit.
        scale_factor (float): Scaling factor for the gradient and hessian.
        initial_params_list (Optional[list]): List of initial parameters for optimization.
        mu_param_dict (Optional[dict]): Dictionary of parameter estimates.
        wald_against_null_results (Optional[dict]): Results of the Wald test.
        bayes_idata (Optional[az.InferenceData]): Inference data for Bayesian analysis.
        num_cores (int): Number of CPU cores to use for parallel processing.
    """

    y: ArrayLike
    X: Optional[ArrayLike] = Field(default=None)
    name: Optional[str] = Field(default=None)

    covariates: Optional[dict] = Field(default_factory=dict)
    covariate_names: list = Field(default=["!Intercept"])

    best_fit_res: OptimizeResult = Field(default=None)
    best_params: list = Field(default=None)
    best_loglik: float = Field(default=np.inf)
    best_initialization: list = Field(default=None)

    scale_factor: Optional[float] = Field(default=float(1e-2))
    initial_params_list: Optional[list] = Field(default_factory=list)

    mu_param_dict: dict = Field(default=None)
    wald_against_null_results: [dict] = Field(default=None)
    bayes_idata: Optional[az.InferenceData] = Field(default=None)
    num_cores: int = Field(default=min(6, os.cpu_count() - 1))
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    has_mle: bool = Field(default=False)

    @field_validator("y", mode="before")
    def validate_y(cls, v):
        """
        Validates and converts the target variable y to a tensor.

        Args:
            v: The target variable.

        Returns:
            A flattened tensor of y.

        Raises:
            ValueError: If y has fewer than 2 unique non-zero values.
        """
        v = cls._convert_to_tensor(v)
        unique_non_zero_values = torch.unique(v[v != 0])
        if len(unique_non_zero_values) < 2:
            raise ValueError(
                "y must have more than 2 non-zero values that aren't the same"
            )
        return v

    @staticmethod
    def _convert_to_tensor(v):
        """
        Converts the input to a tensor.

        Args:
            v: The input array.

        Returns:
            A tensor representation of the input array.

        Raises:
            ValueError: If the input is not a numpy array.
        """
        if v is None:
            return None
        if isinstance(v, list):
            v = np.array(v)
        if isinstance(v, pd.Series):
            v = v.values
        if not isinstance(v, np.ndarray):
            raise ValueError(f"{v} is not a numpy array")
        v = v.ravel()
        v = torch.tensor(v, dtype=torch.float64)
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self.__post_init__()

    def __post_init__(self):
        """
        Post-initialization process to set up the model.
        """
        
        os.environ["PYTENSOR_FLAGS"] = "optimizer_excluding=constant_folding"

        covariate_dummies_list = []
        num_dummy_counts = ceil(0.01 * len(self.y))

        if self.covariates:
            for covariate_name, covariate_data in self.covariates.items():
                covariate_dummies = pd.get_dummies(
                    pd.DataFrame(
                        self.covariates[covariate_name], columns=[covariate_name]
                    ),
                    drop_first=False,
                ).astype(float)

                if covariate_dummies is not None:
                    self.covariate_names += covariate_dummies.columns.tolist()
                    covariate_dummies_list.append(covariate_dummies)

        self.covariate_names = sorted(list(set(self.covariate_names)))
        
        if self.X is None:
            self.X = self._stack_arrays(covariate_dummies_list)
            self.X = sm.add_constant(self.X)
            self.X = torch.tensor(self.X.values, dtype=torch.float64)
        
        self.initial_params_list = self.get_initial_params()
        if not self.initial_params_list:
            raise ValueError("initial_params_list cannot be empty")

    @staticmethod
    def _stack_arrays(arrays):
        """
        Stacks a list of arrays.

        Args:
            arrays: List of arrays to stack.

        Returns:
            A stacked tensor.
        """
        arrays_to_stack = [pd.DataFrame(array) for array in arrays if array is not None]
        
        if arrays_to_stack:
            return pd.concat(arrays_to_stack, axis=1)
        return (
            pd.DataFrame(np.ones((len(arrays[0]), 1)), columns=["!Intercept"])
            if arrays and arrays[0] is not None
            else None
        )
        
    def estimate_with_mle(self):
        """
        Abstract method to get initial parameters for optimization.

        Returns:
            List of initial parameters.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        if not has_mle:
            raise ValueError("MLE is not available")
 
        raise NotImplementedError("Subclasses should implement this method")
        

    def get_initial_params(self):
        """
        Abstract method to get initial parameters for optimization.

        Returns:
            List of initial parameters.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def loglik(self, params, params_tensor = None, ret_torch=False):
        """
        Abstract method to calculate the log-likelihood.

        Args:
            params: Model parameters.
            ret_torch: Whether to return a torch tensor.

        Returns:
            Log-likelihood value.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def gradient(self, params, params_tensor=None):
        """
        Calculates the gradient of the log-likelihood.

        Args:
            params: Model parameters.

        Returns:
            Gradient tensor.
        """
        params = params if params_tensor is None else params_tensor
        log_likelihood = self.loglik(params, ret_torch=True)
        log_likelihood.backward()
        grad = params.grad
        return grad.detach() * self.scale_factor

    def hessian(self, params, params_tensor=None, regularization_params=None):
        """
        Calculates the Hessian matrix of the log-likelihood.

        Args:
            params: Model parameters.
            regularization_params: Regularization parameters.

        Returns:
            Hessian matrix.
        """
        params = params if params_tensor is None else params_tensor
        hessian_matrix = torch.autograd.functional.hessian(self.loglik, params)

        if regularization_params:
            hessian_matrix = self._apply_regularization(
                hessian_matrix, regularization_params
            )

        return hessian_matrix.detach() * (self.scale_factor**2)

    def _apply_regularization(self, hessian_matrix, regularization_params):
        """
        Applies regularization to the Hessian matrix.

        Args:
            hessian_matrix: Hessian matrix.
            regularization_params: Regularization parameters.

        Returns:
            Regularized Hessian matrix.

        Raises:
            ValueError: If the regularization method is invalid.
        """
        method = regularization_params.get("method")
        if method == "ridge":
            return (
                hessian_matrix
                + np.eye(hessian_matrix.shape[0]) * regularization_params["lambda"]
            )
        if method == "norm":
            return hessian_matrix + np.eye(hessian_matrix.shape[0]) * (
                regularization_params["lambda"] * np.linalg.norm(hessian_matrix)
            )
        if method == "norm_constant":
            hessian_matrix += np.eye(hessian_matrix.shape[0]) * (
                regularization_params["lambda"] * np.linalg.norm(hessian_matrix)
            )
            return (
                hessian_matrix
                + np.eye(hessian_matrix.shape[0]) * regularization_params["constant"]
            )
        if method == "eigenvalue":
            eigenvalues, eigenvectors = np.linalg.eigh(hessian_matrix)
            eigenvalues = np.where(
                eigenvalues < regularization_params["threshold"],
                regularization_params["threshold"],
                eigenvalues,
            )
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        raise ValueError(
            "Invalid regularization method. Choose 'ridge', 'norm', 'norm_constant', or 'eigenvalue.'"
        )

    def _run_minimize(
        self,
        params_init,
        optimization_method,
        regularization_params=None,
        verbose=False,
        safe=False,
        minimize_options=None,
    ):
        """
        Runs the optimization to minimize the log-likelihood.

        Args:
            params_init: Initial parameters for optimization.
            optimization_method: Optimization method.
            regularization_params: Regularization parameters.
            verbose: Whether to print optimization details.
            safe: Whether to catch and handle optimization errors.
            minimize_options: Options for the minimize function.

        Returns:
            Tuple of optimized parameters, log-likelihood, success status, and message.
        """
        if not np.all(np.isfinite(params_init)):
            print(f"Invalid initial parameters: {params_init}")
            return None, np.inf, False, "Invalid initial parameters"

        params_tensor = torch.tensor(params_init, dtype=torch.float64, requires_grad=True)
        gradient_fn = lambda params, *args: self.gradient(params, params_tensor=params_tensor)
        hessian_fn = (
            lambda params, *args: self.hessian(params, params_tensor=params_tensor, regularization_params=regularization_params)
            if optimization_method not in ["BFGS", "L-BFGS-B"]
            else None
        )

        try:
            result = minimize(
                self.loglik,
                params_init,
                method=optimization_method,
                jac=gradient_fn,
                hess=hessian_fn,
                args=(params_tensor, False),
                options=minimize_options,
            )
            if verbose:
                print(result)
            return result.x, result.fun, result.success, result.message
        except ValueError as e:
            if safe:
                print(f"Optimization error with params {params_init}: {e}")
                return None, np.inf, False, str(e)
            else:
                raise e

    def _parallel_optimization(
        self,
        optimization_method,
        verbose,
        regularization_params=None,
        minimize_options=None,
    ):
        """
        Runs parallel optimization using multiple initial parameters.

        Args:
            optimization_method: Optimization method.
            verbose: Whether to print optimization details.
            regularization_params: Regularization parameters.
            minimize_options: Options for the minimize function.
        """
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            futures = {
                executor.submit(
                    self._run_minimize,
                    params,
                    optimization_method=optimization_method,
                    verbose=verbose,
                    regularization_params=regularization_params,
                ): i
                for i, params in enumerate(self.initial_params_list)
            }
            for future in as_completed(futures):
                self._update_best_fit(future.result(), futures[future])

    def _sequential_optimization(
        self,
        optimization_method,
        verbose,
        regularization_params=None,
        minimize_options=None,
    ):
        """
        Runs sequential optimization using multiple initial parameters.

        Args:
            optimization_method: Optimization method.
            verbose: Whether to print optimization details.
            regularization_params: Regularization parameters.
            minimize_options: Options for the minimize function.
        """
        for i, params_init in enumerate(self.initial_params_list):
            self._update_best_fit(
                self._run_minimize(
                    params_init,
                    optimization_method=optimization_method,
                    verbose=verbose,
                    regularization_params=regularization_params,
                ),
                i,
            )

    def fit(
        self,
        optimization_method="dogleg",
        verbose=False,
        parallel=False,
        ret_time=False,
        regularization_params=None,
        safe=False,
        minimize_options={"disp": False, "maxiter": 5000},
    ):
        """
        Fits the model using the specified optimization method.

        Args:
            optimization_method: Optimization method.
            verbose: Whether to print optimization details.
            parallel: Whether to run optimization in parallel.
            ret_time: Whether to return the fitting time.
            regularization_params: Regularization parameters.
            safe: Whether to catch and handle optimization errors.
            minimize_options: Options for the minimize function.

        Returns:
            Best parameters, log-likelihood, best initialization, and optionally the fitting time.

        Raises:
            ValueError: If no initialization converged.
        """
        start_time = time.time()
        
        if self.has_mle:
            self.best_params = self.estimate_with_mle()
            self.best_loglik = self.loglik(self.best_params)
            self.best_initialization = "MLE"
            self.best_fit_res = "MLE"
            
        if parallel:
            self._parallel_optimization(
                optimization_method, verbose, regularization_params, minimize_options
            )
        else:
            self._sequential_optimization(
                optimization_method, verbose, regularization_params, minimize_options
            )

        if self.best_params is None:
            raise ValueError("No initialization converged")

        num_betas = len(self.covariate_names) + 1
        params_est_beta = self.best_params[: num_betas + 1]
        self.mu_param_dict = dict(zip(self.covariate_names, params_est_beta.tolist()))

        if ret_time:
            return (
                self.best_params,
                self.best_loglik,
                self.best_initialization,
                time.time() - start_time,
            )

        return self.best_params, self.best_loglik, self.best_initialization

    def _update_best_fit(self, res, i):
        """
        Updates the best fit parameters if the current result is better.

        Args:
            res: Result of the optimization.
            i: Index of the initial parameters.
        """
        params_new, loglik_new, success, message = res
        if loglik_new < self.best_loglik:
            self.best_loglik = loglik_new
            self.best_params = params_new
            self.best_initialization = self.initial_params_list[i]
            self.best_fit_res = res
            
    @staticmethod
    def _get_linear_estimator(params, size, num_covariates, design_matrix=None):
        """
        Prepare the design matrix and calculate mu.

        Parameters:
        - params: array-like, parameters of the model.
        - size: int, number of samples or the size of the range.
        - num_covariates: int, number of covariates.
        - design_matrix: array-like, optional, design matrix for the covariates.

        Returns:
        - eta: array, linear predictor.
        """
        betas_with_intercept = params[:num_covariates]

        if design_matrix is None:
            design_matrix = np.zeros((size, num_covariates))
            design_matrix[:, 0] = 1

            # Distribute ones across the rest of the columns evenly
            for i in range(1, num_covariates):
                start_idx = (i - 1) * (size // num_covariates)
                end_idx = i * (size // num_covariates)
                design_matrix[start_idx:end_idx, i] = 1
        elif design_matrix.shape[0] != size:
            # Ensure the proportions are exactly the same
            repeats = size // design_matrix.shape[0]
            remainder = size % design_matrix.shape[0]
            design_matrix = np.vstack([design_matrix] * repeats + [design_matrix[:remainder]])

        eta = np.dot(design_matrix, np.exp(betas_with_intercept))
        return eta

    def empirical_bayes_inference(self, fixed_sigma=1.0, num_samples=500, tune=1000):
        """
        Abstract method for empirical Bayes inference.

        Args:
            fixed_sigma: Fixed sigma value.
            num_samples: Number of samples.
            tune: Tuning parameter.

        Returns:
            Inference results.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def wald_test_against_null(self, verbose=False):
        """
        Performs Wald test against the null hypothesis.

        Args:
            verbose: Whether to print details.

        Returns:
            Dictionary of Wald statistics and p-values.
        """
        trace = self.empirical_bayes_inference()

        results = {}
        posterior = trace.posterior

        if "beta" in posterior.data_vars:
            beta_samples = posterior["beta"].values
            for i in range(1, beta_samples.shape[-1]):
                samples = beta_samples[..., i].flatten()

                mean_estimate = np.mean(samples)
                std_error = np.std(samples)

                wald_stat = (mean_estimate / std_error) ** 2
                p_value = chi2.sf(wald_stat, df=1)

                results[f"{self.covariate_names[i]}"] = [wald_stat, p_value]

        self.wald_against_null_results = results

        return results
