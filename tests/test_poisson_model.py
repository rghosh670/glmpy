import torch
import numpy as np
import glmpy
from glmpy import PoissonModel
import pytest

def generate_data(size, poisson_lambdas, probabilities):
    # Define the probabilities and Poisson parameters for each cell type
    cell_types = ["Cardiomyocytes", "Ectoderm", "Pluripotent"]
    
    cell_type_samples = np.random.choice(cell_types, size=size, p=probabilities)
    
    # Generate y values based on cell type
    y_samples = np.zeros(size, dtype=int)
    
    for i, cell_type in enumerate(cell_types):
        indices = np.where(cell_type_samples == cell_type)[0]
        y_samples[indices] = np.random.poisson(lam=poisson_lambdas[i], size=len(indices))
    
    covariates = {
        "cell_type": cell_type_samples,
    }
    
    return y_samples, covariates

@pytest.fixture(params=[
    # Basic case with well-separated means
    {'size': 400, 'poisson_lambdas': [13, 50, 200], 'probabilities': [0.4, 0.2, 0.4]},
    # Small sample size
    {'size': 100, 'poisson_lambdas': [13, 50, 200], 'probabilities': [0.4, 0.2, 0.4]},
    # Large sample size
    # {'size': 100000, 'poisson_lambdas': [13, 50, 200], 'probabilities': [0.4, 0.2, 0.4]},
    # Very high lambda values
    {'size': 1000, 'poisson_lambdas': [130, 500, 2000], 'probabilities': [0.4, 0.2, 0.4]},
    # Very low lambda values
    {'size': 1000, 'poisson_lambdas': [1, 2, 3], 'probabilities': [0.4, 0.2, 0.4]},
    # Varying probabilities
    {'size': 1000, 'poisson_lambdas': [13, 50, 200], 'probabilities': [0.1, 0.3, 0.6]},
    # Large lambda values with high probabilities
    {'size': 1000, 'poisson_lambdas': [1000, 2000, 5000], 'probabilities': [0.6, 0.3, 0.1]},
    # Small lambda values with low probabilities
    {'size': 1000, 'poisson_lambdas': [0.5, 1, 2], 'probabilities': [0.8, 0.1, 0.1]},
    # Mixed lambda values with equal probabilities
    {'size': 1000, 'poisson_lambdas': [5, 500, 50], 'probabilities': [0.33, 0.33, 0.34]},
    # Large sample size with large lambda values
    {'size': 10000, 'poisson_lambdas': [100, 300, 500], 'probabilities': [0.5, 0.25, 0.25]},
    # Small sample size with mixed lambda values
    {'size': 50, 'poisson_lambdas': [10, 100, 1000], 'probabilities': [0.4, 0.4, 0.2]},
    # Very high lambda values with varying probabilities
    {'size': 1000, 'poisson_lambdas': [1000, 2000, 3000], 'probabilities': [0.1, 0.2, 0.7]},
    # Very low lambda values with high probabilities
    {'size': 1000, 'poisson_lambdas': [0.1, 0.2, 0.3], 'probabilities': [0.7, 0.2, 0.1]},
    # # Large sample size with very low lambda values
    # # {'size': 50000, 'poisson_lambdas': [0.5, 1, 1.5], 'probabilities': [0.3, 0.4, 0.3]},
    # Small sample size with very high lambda values
    {'size': 100, 'poisson_lambdas': [500, 1000, 2000], 'probabilities': [0.2, 0.3, 0.5]},
    # Mixed probabilities with varied lambda values
    {'size': 1000, 'poisson_lambdas': [20, 200, 2000], 'probabilities': [0.25, 0.25, 0.5]},
])
def data(request):
    params = request.param
    y, covariates = generate_data(size=params['size'], poisson_lambdas=params['poisson_lambdas'], probabilities=params['probabilities'])
    return y, covariates, np.log(params['poisson_lambdas'])

def test_poisson_model(data):
    y, covariates, log_poisson_lambdas = data
    model = PoissonModel(y=y, covariates=covariates, name="PoissonModel")

    fit_res = model.fit(
        optimization_method="dogleg",
        verbose=False,
        parallel=False,
        ret_time=False,
        regularization_params=None,
        safe=False,
        minimize_options={"disp": False, "maxiter": None},
    )

    params = fit_res[0]
    
    params_tensor = torch.tensor(params).reshape(-1, 1)  # Ensure it's a column vector
    log_poisson_lambdas_tensor = torch.tensor(np.hstack((np.log(np.mean(y)), log_poisson_lambdas))).reshape(-1, 1)  # Ensure it's a column vector
    
    final_loglikelihood = model.loglik(params_tensor)
    theoretical_loglikelihood = model.loglik(log_poisson_lambdas_tensor)
    
    loglikelihood_diff = -final_loglikelihood + theoretical_loglikelihood
    
    print(f"Log-likelihood difference for theoretical means and final parameters: {loglikelihood_diff}")
    
    for param, log_lambda in zip(params[1:], log_poisson_lambdas):
        if np.abs(log_lambda) < 0.2:
            assert -1 <= param <= 1, f"Param {param} not within [-1, 1] for log_mu {log_lambda}"
        else:
            assert np.abs(param - log_lambda) <= np.abs(0.3 * log_lambda), f"Param {param} not within acceptable range of {log_lambda}"

if __name__ == "__main__":
    # pytest.main(["-k", "data[0]"])  # Run only the first test case
    pytest.main()  # Run all test cases
