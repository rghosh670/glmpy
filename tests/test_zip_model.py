import torch
import numpy as np
import glmpy
from glmpy import ZIPModel
import pytest
from scipy.special import expit, logit
from icecream import ic

def generate_zip_data(size, mus, p0, probabilities):
    # Define the probabilities and ZIP parameters for each cell type
    cell_types = ["Cardiomyocytes", "Ectoderm", "Pluripotent"]
    
    cell_type_samples = np.random.choice(cell_types, size=size, p=probabilities)
    
    # Generate y values based on cell type
    y_samples = np.zeros(size, dtype=int)
    
    pi = expit(p0)
    
    for i, cell_type in enumerate(cell_types):
        indices = np.where(cell_type_samples == cell_type)[0]
        mu = mus[i]
        
        for idx in indices:
            if np.random.rand() < pi:
                y_samples[idx] = 0
            else:
                y_samples[idx] = np.random.poisson(mu)
    
    covariates = {
        "cell_type": cell_type_samples,
    }
    
    return y_samples, covariates

@pytest.fixture(params=[
    # Basic case with well-separated means
    {'size': 1000, 'mus': [13, 50, 200], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
    # Small sample size
    {'size': 1000, 'mus': [13, 50, 200], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
    # Large sample size
    {'size': 1000, 'mus': [13, 50, 200], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
    # Very high mu values
    {'size': 1000, 'mus': [130, 500, 2000], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
    # Very low mu values
    {'size': 1000, 'mus': [1, 2, 3], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
    # Varying probabilities
    {'size': 1000, 'mus': [13, 50, 200], 'p0': logit(0.3), 'probabilities': [0.4, 0.2, 0.4]},
])
def data(request):
    params = request.param
    y, covariates = generate_zip_data(size=params['size'], mus=params['mus'], p0=params['p0'], probabilities=params['probabilities'])
    return y, covariates, np.log(params['mus']), params['p0']

def test_zip_model(data):
    y, covariates, log_mus, p0 = data
    model = ZIPModel(y=y, covariates=covariates, name="ZIPModel")

    fit_res = model.fit(
        optimization_method="trust-constr",
        verbose=False,
        parallel=False,
        ret_time=False,
        regularization_params=None,
        safe=False,
        minimize_options={"disp": False, "maxiter": None},
    )

    params = fit_res[0]
    fitted_log_mus = params[1:len(log_mus) + 1]
    fitted_p0 = params[-1]
    
    params_tensor = torch.tensor(params).reshape(-1, 1)  # Ensure it's a column vector
    pred_tensor = torch.tensor(np.hstack((np.log(np.mean(y[y!=0])), log_mus, p0))).reshape(-1, 1)  # Ensure it's a column vector
    
    final_loglikelihood = model.loglik(params_tensor)
    theoretical_loglikelihood = model.loglik(pred_tensor)
    
    loglikelihood_diff = -final_loglikelihood + theoretical_loglikelihood
    
    print(f"Log-likelihood difference for theoretical means and final parameters: {loglikelihood_diff}")

    for param, log_mu in zip(fitted_log_mus, log_mus):
        if np.abs(log_mu) < 0.2:
            assert -1 <= param <= 1, f"Log_mu {param} not within [-1, 1] for log_mu {log_mu}"
        else:
            assert np.abs(param - log_mu) <= np.abs(0.3 * log_mu), f"Log_mu {param} not within acceptable range of {log_mu}"
    
    assert np.abs(fitted_p0 - p0) <= abs(0.3 * p0), f"p0 {fitted_p0} not within acceptable range of {p0}"

if __name__ == "__main__":
    pytest.main()
