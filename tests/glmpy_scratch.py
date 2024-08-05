import numpy as np
import polars as pl
import pandas as pd
import glmpy
from glmpy import NBModel, PoissonModel, ZINBModel, ZIPModel

def generate_data(size=1000, mus=[13, 50, 200], alphas=[1.5, 2.5, 3.5], probabilities=[0.4, 0.2, 0.4]):
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


def main():
    y, covariates = generate_data(size = 1000)

    model = NBModel(y = y, covariates = covariates, name = "NBModel")

    fit_res = model.fit(
        use_mle=False,
        optimization_method="dogleg",
        verbose=False,
        parallel=False,
        ret_time=False,
        regularization_params=None,
        safe=False,
        minimize_options={"disp": False, "maxiter": 5000},
    )

    ic(model.best_fit_res)


if __name__ == "__main__":
    main()
