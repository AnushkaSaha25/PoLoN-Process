# PoLoN-Process
Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data
# Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data

### Overview
Modeling datasets of integer counts is crucial in physics and other scientific disciplines, where measurements involve discrete, non-negative quantities.  
Traditional approaches such as Poisson regression often struggle to capture complex, non-linear relationships.  
The **Poisson Log-Normal (PoLoN) process** combines the flexibility of Gaussian Processes with the Poisson-lognormal distribution to accurately model integer count data.

This repository contains the implementation, synthetic data experiments, and real-world applications (e.g., Higgs Boson signal analysis and bike rental prediction) for the PoLoN framework.

---

### Key Features
- Non-parametric model for count data prediction  
- Combines Poisson likelihood with Gaussian Process prior  
- Handles discrete, non-negative data  
- Captures correlations through kernel-based inference  
- Supports both synthetic and real-world datasets  

---

### Setup
To run the project, it is recommended to use Python 3.9 or higher.  
Install all required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
Then, launch the Jupyter notebook:

```bash
jupyter notebook PoLoN.ipynb
```

---


## Usage

The main function provided for future reference in this repository is `polon_predict_and_plot`, which performs **training, prediction, and plotting** of count data using the PoLoN process.

### Function Overview

**`polon_predict_and_plot`**:

- Trains the PoLoN model on input training data `(X_train, t_train)`.
- Optimizes kernel hyperparameters using the log-likelihood function.
- Predicts the expected counts (Poisson mean) and the most probable Poisson outputs (mode) for new input points.
- Generates plots showing predictive mean, 2-sigma confidence intervals, and most probable outputs.

This function internally uses several helper functions defined in `helpers.py`:

- **`compute_cov_matrix`**: Computes the RBF covariance matrix for the training points.  
- **`compute_K`**: Computes the covariance vector between training points and a new input point.  
- **`neg_log_likelihood_function`**: Computes the negative log-likelihood for hyperparameter optimization.  
- **`newton_optimization`**: Solves for the latent variables (`lambda`) using Newton’s method.  
- **`rbf_kernel_extended`**: Computes the RBF kernel value between two points.  

Other helper functions are used internally for computing Hessians, Cholesky decompositions, Gaussian bumps, and log-likelihood contributions.

### Inputs

| Parameter      | Type       | Description |
|----------------|------------|-------------|
| `X_train`      | `np.ndarray` | Training input features (1D or 2D) |
| `t_train`      | `np.ndarray` | Training output counts |
| `X_input`      | `np.ndarray`, optional | Points at which to predict outputs. Default: 100 evenly spaced points in training range |
| `theta_init`   | `np.ndarray`, optional | Initial guess for kernel hyperparameters. Default: `[0.01, 20.0]` |
| `bounds`       | list of tuples, optional | Bounds for hyperparameters in optimization. Default: `[(0.0001, 30), (0.01, 30)]` |
| `n_samples`    | int, optional | Number of Monte Carlo samples to compute the most probable Poisson outputs. Default: 5000 |

### Outputs

Returns a dictionary containing:

- `X_input`: Array of input points where predictions were made  
- `mu_values`: Predictive mean of the latent log-lambda at each `X_input`  
- `std_values`: Predictive standard deviation of the latent log-lambda  
- `poisson_output`: Predictive mean of Poisson counts (expected counts)  
- `lower_bound`, `upper_bound`: 2-sigma confidence intervals for predicted Poisson counts  
- `most_probable_output`: Most probable Poisson counts (mode) for each `X_input`  

Plots showing predictive mean, confidence intervals, and most probable outputs are also generated.

### Example

```python
from helpers import polon_predict_and_plot
import numpy as np

# Example training data
X_train = np.array([0.1, 0.12, 0.13, 0.15])
t_train = np.array([5, 8, 12, 15])

# Run PoLoN prediction
results = polon_predict_and_plot(X_train, t_train)

# Access predicted Poisson mean and most probable outputs
poisson_mean = results["poisson_output"]
most_probable = results["most_probable_output"]

print("Predicted Poisson mean:", poisson_mean)
print("Most probable Poisson outputs:", most_probable)
---
### Repository Structure
```
PoLoN-Process/
│
├── PoLoN.ipynb              # Main Jupyter notebook with training, inference, and plots
├── data/                     # Example datasets (synthetic + real)
│   ├── synthetic.csv
│   └── higgs.csv
├── helper_functions.py       # Supporting Python functions used by the notebook
├── results/                  # Generated output figures and result summaries
├── requirements.txt          # List of required Python packages
└── README.md                 # This file
```
### 🧩 Examples

```bash
python generate_synthetic_data.py


