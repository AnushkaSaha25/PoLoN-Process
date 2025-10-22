# PoLoN-Process
Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data
# Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data

### Overview
Modeling datasets of integer counts is crucial in physics and other scientific disciplines, where measurements involve discrete, non-negative quantities. Traditional approaches such as Poisson regression often struggle to capture complex, non-linear relationships.  

The **Poisson Log-Normal (PoLoN) process** combines the flexibility of Gaussian Processes with the Poisson-lognormal distribution to model integer count data accurately.  

This repository provides:  
- Core implementations of PoLoN for predicting expected counts and most probable outcomes.  
- A variant of the prediction function that incorporates **signal + background decomposition**, useful for applications like particle physics where separating signal from background noise is important.  
- Synthetic experiments and real-world examples, such as Higgs Boson signal analysis and bike rental demand prediction.  

---

### Key Features
- Non-parametric model for count data prediction  
- Combines Poisson likelihood with Gaussian Process prior  
- Handles discrete, non-negative data  
- Captures correlations through kernel-based inference  
- Supports both synthetic and real-world datasets  

---
### Repository Structure
```
PoLoN-Process/
│
├── PoLoN.ipynb              # Main Jupyter notebook with training, inference, and plots
├── data/                     # Example datasets (synthetic + real)
│   ├── synthetic.csv
│   └── higgs.csv
├── helpers.py       # Supporting Python functions used by the notebook
├── requirements.txt          # List of required Python packages
└── README.md                 # This file
```
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


### Usage

This repository provides tools for **training, prediction, and visualization** of count data using the PoLoN (Poisson Log-Normal) process.  
The main workflow is implemented through two functions — **`polon_predict_and_plot`** and **`predict_signal_background`** — both of which leverage helper functions defined in `helpers.py`.

---

#### Helper Function Reference

These helper functions, defined in `helpers.py`, support the main PoLoN prediction workflow:

- **`compute_cov_matrix`**: Computes the RBF covariance matrix for the training points.  
- **`compute_K`**: Computes the covariance vector between training points and a new input point.  
- **`neg_log_likelihood_function`**: Computes the negative log-likelihood for hyperparameter optimization.  
- **`newton_optimization`**: Solves for the latent variables (`lambda`) using Newton’s method.  
- **`rbf_kernel_extended`**: Computes the RBF kernel value between two points.  

Other internal utilities handle Hessians, Cholesky decompositions, Gaussian bumps, and log-likelihood contributions.

---

#### Function 1: `polon_predict_and_plot`

This function models integer count data using the **PoLoN framework** without explicitly separating signal and background components.  

##### **Function Overview**

- Trains the PoLoN model on input training data `(X_train, t_train)`.
- Optimizes kernel hyperparameters via log-likelihood maximization.
- Predicts the **expected counts** (Poisson mean) and **most probable counts** (Poisson mode) for new input points.
- Computes **Monte Carlo-based 95% confidence intervals** for predicted counts.
- Generates plots showing the predictive mean, confidence intervals, and most probable outputs.
> **Note:** The main functions are defined in `helpers.py` for future reuse and do not appear directly in the Jupyter notebook.  
> This modular structure allows users to reuse functions without modifying notebook code.

##### **Inputs**

| Parameter      | Type       | Description |
|----------------|------------|-------------|
| `X_train`      | `np.ndarray` | Training input features (1D or 2D array) |
| `t_train`      | `np.ndarray` | Training output counts |
| `X_input`      | `np.ndarray`, optional | Points at which to predict outputs. Default: 100 evenly spaced points across the training range |
| `theta_init`   | `np.ndarray`, optional | Initial guess for kernel hyperparameters. Default: `[0.01, 20.0]` |
| `bounds`       | list of tuples, optional | Bounds for hyperparameters during optimization. Default: `[(0.0001, 30), (0.01, 30)]` |
| `n_samples`    | int, optional | Number of Monte Carlo samples for most probable Poisson outputs. Default: 5000 |

##### **Outputs**

- `X_input`: Input points where predictions were made  
- `mu_values`: Predictive mean of the latent log-rate  
- `std_values`: Predictive standard deviation of the latent log-rate  
- `poisson_mean_output`: Predictive mean of Poisson counts  
- `lower_bounds`, `upper_bounds`: Monte Carlo-based 95% confidence intervals  
- `most_probable_output`: Most probable Poisson counts (mode)  

Plots showing the predictive mean, confidence intervals, and most probable outputs are automatically generated.

##### **Example**

```python
from helpers import *
import numpy as np

# Example training data
X_train = np.array([0.1, 0.12, 0.13, 0.15])
t_train = np.array([5, 8, 12, 15])

# Run PoLoN prediction
results = polon_predict_and_plot(X_train, t_train)

# Access predicted results
poisson_mean = results["poisson_mean_output"]
most_probable = results["most_probable_output"]
lower_ci = results["lower_bounds"]
upper_ci = results["upper_bounds"]

print("Predicted Poisson mean:", poisson_mean)
print("Most probable Poisson outputs:", most_probable)
print("95% confidence intervals:", list(zip(lower_ci, upper_ci)))

```
#### Function 2: `predict_signal_background`

This function extends the PoLoN framework to **jointly model background and signal components** in count data.  
It is particularly useful in **physics analyses** (e.g., Higgs Boson detection), where observed data often consist of a smooth **background** plus a localized **signal** peak.

##### **Function Overview**

- Separately optimizes **PoLoN hyperparameters** on the background component.
- Fits a **Gaussian signal model** on the signal region, allowing the amplitude, position, and width of the signal to vary freely.
- Predicts the **combined PoLoN + Gaussian** count distribution across the full data range.
- Produces decomposed outputs for background, signal, and total count prediction.

The function ensures a clear distinction between **background** (PoLoN process) and **signal** (Gaussian bump), maintaining physical interpretability.

##### **Inputs**

| Parameter              | Type           | Description |
|------------------------|----------------|-------------|
| `X_bg`                 | `np.ndarray`   | Background input points |
| `t_bg`                 | `np.ndarray`   | Observed background counts |
| `X_signal`             | `np.ndarray`   | Signal input points |
| `t_signal`             | `np.ndarray`   | Observed signal counts |
| `bounds_theta`         | list of tuples, optional | Bounds for PoLoN hyperparameters. Default: `[(0.0001, 2), (0.01, 50)]` |
| `theta_init`           | `np.ndarray`, optional | Initial guess for PoLoN hyperparameters. Default: `[0.05, 20.0]` |
| `bounds_gauss`         | list of tuples, optional | Bounds for Gaussian signal parameters (scaled). Default: `[(0.01, 10), (μ_min/0.1, μ_max/0.1), (1.0, 50.0)]` |
| `trans_params0_scaled` | list or `np.ndarray`, optional | Initial guess for scaled Gaussian parameters. Default: `[1.0, mean(X_signal)/0.1, 0.01]` |
| `X_input`              | `np.ndarray`, optional | Points at which predictions are made. Default: 100 points spanning all data |

##### **Outputs**

The function returns a dictionary containing:

- `theta_opt`: Optimized PoLoN hyperparameters  
- `A_opt`: Optimized Gaussian amplitude (signal strength)  
- `mu_opt`: Optimized Gaussian mean (signal position)  
- `sigma_opt`: Optimized Gaussian width (signal spread)  
- `X_input`: Prediction points across the full data range  
- `mu_values`: Predicted PoLoN latent mean (background)  
- `std_values`: Predicted PoLoN latent standard deviation  
- `gaussian_signal`: Fitted Gaussian signal contribution  
- `poisson_mean`: Combined PoLoN + Gaussian mean counts  
- `theta_success`, `gauss_success`: Optimization success flags  

> **Note:**  
> This function can be computationally expensive, especially when exploring multiple signal strengths or realizations.  
> For large-scale experiments, **parallelization (e.g., via Amarel job submission or HPC clusters)** is recommended.  
> The function itself is defined in `helpers.py` for modular reuse and does not appear directly in the Jupyter notebook.
---




