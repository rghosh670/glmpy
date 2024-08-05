import torch
import numpy as np
import glmpy
from glmpy import NBModel
import pytest

def generate_data(size, mus, alphas, probabilities):
    # Define the probabilities and Negative Binomial parameters for each cell type
    cell_types = ["Cardiomyocytes", "Ectoderm", "Pluripotent"]
    
    cell_type_samples = np.random.choice(cell_types, size=size, p=probabilities)
    
    # Generate y values based on cell type
    y_samples = np.zeros(size, dtype=int)
    
    for i, cell_type in enumerate(cell_types):
        indices = np.where(cell_type_samples == cell_type)[0]
        mu = mus[i]
        alpha = alphas[i]
        r = alpha
        p = r / (r + mu)
        y_samples[indices] = np.random.negative_binomial(n=r, p=p, size=len(indices))
    
    covariates = {
        "cell_type": cell_type_samples,
    }
    
    return y_samples, covariates

@pytest.fixture(params=[
    # Basic case with well-separated means and varied alphas
    {'size': 10000, 'mus': [13, 50, 200], 'alphas': [1.5, 2.5, 3.5], 'probabilities': [0.4, 0.2, 0.4]},
    # High mus, low alphas
    {'size': 10000, 'mus': [130, 500, 2000], 'alphas': [0.5, 1.0, 1.5], 'probabilities': [0.3, 0.3, 0.4]},
    # Low mus, high alphas
    {'size': 10000, 'mus': [1, 2, 3], 'alphas': [5.0, 6.0, 7.0], 'probabilities': [0.5, 0.2, 0.3]},
    # Mixed mus and alphas
    {'size': 10000, 'mus': [20, 500, 50], 'alphas': [1.0, 2.0, 3.0], 'probabilities': [0.25, 0.25, 0.5]},
    # High probabilities for one class
    {'size': 10000, 'mus': [15, 45, 150], 'alphas': [1.0, 1.5, 2.0], 'probabilities': [0.7, 0.2, 0.1]},
    # Low probabilities for one class
    {'size': 10000, 'mus': [25, 75, 250], 'alphas': [2.0, 2.5, 3.0], 'probabilities': [0.1, 0.4, 0.5]},
    # High alphas with varying probabilities
    {'size': 10000, 'mus': [30, 80, 300], 'alphas': [4.0, 5.0, 6.0], 'probabilities': [0.3, 0.3, 0.4]},
])

def data(request):
    params = request.param
    y, covariates = generate_data(size=params['size'], mus=params['mus'], alphas=params['alphas'], probabilities=params['probabilities'])
    return y, covariates, np.log(params['mus']), params['alphas']

def test_nb_model(data):
    y, covariates, log_mus, alphas = data
    model = NBModel(y=y, covariates=covariates, name="NBModel")

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
    
    params_tensor = torch.tensor(params).reshape(-1, 1)  # Ensure it's a column vector
    
    log_mus_pred_tensor = torch.tensor(np.hstack((np.log(np.mean(y)), log_mus))).reshape(-1, 1)  # Ensure it's a column vector
    alpha_pred_tensor = torch.tensor(np.hstack(((np.mean(y) ** 2) / (np.var(y) - np.mean(y)), alphas))).reshape(-1, 1)  # Ensure it's a column vector
    pred_tensor = torch.cat((log_mus_pred_tensor, alpha_pred_tensor))
    
    final_loglikelihood = model.loglik(params_tensor)
    theoretical_loglikelihood = model.loglik(pred_tensor)
    
    loglikelihood_diff = -final_loglikelihood + theoretical_loglikelihood
    
    print(f"Log-likelihood difference for theoretical means and final parameters: {loglikelihood_diff}")
    
    for param, log_mu in zip(params[1:], log_mus):
        if np.abs(log_mu) < 0.2:
            assert -1 <= param <= 1, f"Param {param} not within [-1, 1] for log_mu {log_mu}"
        else:
            assert np.abs(param - log_mu) <= np.abs(0.3 * log_mu), f"Param {param} not within acceptable range of {log_mu}"
        
    # for param, alpha in zip(params[1:], alphas):
    #     assert np.abs(param - alpha) <= np.abs(0.5 * alpha), f"Alpha {param} not within acceptable range of {alpha}"

if __name__ == "__main__":
    pytest.main()
