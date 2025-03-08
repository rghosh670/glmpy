## Overview

`glmpy` is a high-performance Python package for efficient modeling of Generalized Linear Models (GLMs). It provides a user-friendly interface for fitting, analyzing, and evaluating GLMs with a focus on speed and scalability for both small and large datasets.

## Installation

Install the latest stable version from PyPI:

```bash
pip install glmpy
```

For development version:

```bash
git clone https://github.com/rghosh670/glmpy.git
cd glmpy
pip install -e .
```

## Features

- Fast implementation of common GLM families (Gaussian, Binomial, Poisson, Gamma, etc.)
- Support for various link functions (identity, log, logit, probit, etc.)
- Efficient parameter estimation using multiple optimization methods
- Automated feature selection capabilities
- Comprehensive model diagnostics and goodness-of-fit measures
- Integration with scikit-learn pipelines
- GPU acceleration for large datasets (optional)
- Built-in cross-validation and regularization options
- Robust against convergence issues common in GLM implementations
- Extensible architecture for custom distributions and link functions

## Quick Start

```python
import numpy as np
from glmpy import GLM

# Generate some example data
X = np.random.normal(size=(1000, 5))
beta = np.array([0.5, -0.2, 1.0, 0.0, -0.7])
linear_pred = X @ beta
y = np.random.poisson(np.exp(linear_pred))

# Fit Poisson regression with log link
model = GLM(family="poisson", link="log")
model.fit(X, y)

# View model summary
print(model.summary())

# Make predictions on new data
X_new = np.random.normal(size=(10, 5))
predictions = model.predict(X_new)
```

## Documentation

Comprehensive documentation is available at [https://glmpy.readthedocs.io](https://glmpy.readthedocs.io).

This includes:
- API reference
- Tutorials and examples
- Performance benchmarks
- Theoretical background
- Contributing guidelines

## Roadmap

- [ ] Additional distribution families
- [ ] Mixed-effects models extension
- [ ] More optimization algorithms
- [ ] Time-series specific components
- [ ] Advanced diagnostics visualizations

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `glmpy` in your research, please cite:

```
@software{glmpy2025,
  author = {Ghosh, R.},
  title = {glmpy: Fast Generalized Linear Models in Python},
  url = {https://github.com/rghosh670/glmpy},
  version = {x.x.x},
  year = {2025},
}
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
