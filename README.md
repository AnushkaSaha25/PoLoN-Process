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
├── Juputer_nootbook/                      # Main Jupyter notebooks with training, inference, and plots
│   ├── PoLoN_predictive.ipynb             # the predictive methode, equivalent to the function polon_predict_and_plot()
│   └── PoLoN_signal_background.ipynb      # the signal extraction method, equivalent to the function predict_signal_background_with_plot()        
├── data/                     # Example datasets 
│   ├── hour.csv
│   └── unbinned_diphoton_mass.npy
├── helper.py       # Supporting Python functions used by the notebook
├── requirements.txt          # List of required Python packages
└── README.md                 # This file
```
---
### Data Directory and Example Datasets

All example datasets should be placed in the `data/` folder. You can include `.csv` or `.npy` files. Below is a brief description of the current example datasets provided:

| File Name                        | Type   | Description |
|----------------------------------|--------|-------------|
| `hour.csv`                        | CSV    | Hourly bike rental counts (real-world example) |
| `unbinned_diphoton_mass.npy`      | NPY    | Unbinned diphoton mass data used for signal-background PoLoN example (Higgs search) |

**Usage Example:**

```python
import numpy as np

# Load unbinned diphoton mass dataset
data = np.load("data/unbinned_diphoton_mass.npy")

# Load bike rental dataset
import pandas as pd
bike_data = pd.read_csv("data/hour.csv")
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
The main workflow is implemented through two functions — **`polon_predict_and_plot`** and **`predict_signal_background`** — both of which leverage helper functions defined in `helper.py`.

---

#### Helper Function Reference

These helper functions, defined in `helper.py`, support the main PoLoN prediction workflow:

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
> **Note:** The main functions are defined in `helper.py` for future reuse and do not appear directly in the Jupyter notebook.  
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
from helper import *
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
---
#### Function 2: `predict_signal_background_with_plot`

This function models count data using the **PoLoN (Poisson–Lognormal)** process, which inherently represents both **background** and **signal** components within a unified probabilistic framework.  
It automatically fits the model, optimizes parameters, and visualizes the **PoLoN prediction** along with the **signal–background decomposition**.

This approach is particularly relevant in **physics applications** (e.g., Higgs Boson searches), where observed data are composed of a smooth stochastic background and a localized signal peak — both naturally represented within the PoLoN process.


##### **Function Overview**

- Optimizes **PoLoN hyperparameters** using the background-dominated region of the data.  
- Extends the fitted PoLoN model to include a **localized signal variation**, modeled internally through an additional parameterized component.  
- Predicts the full **PoLoN intensity field** (signal + background combined) across the entire input range.  
- Automatically generates two subplots:
  1. Total PoLoN prediction vs observed data  
  2. Signal–background decomposition  


##### **Inputs**

| Parameter | Type | Description |
|------------|------|-------------|
| `X_bg` | `np.ndarray` | Input points for the background region |
| `t_bg` | `np.ndarray` | Observed background counts |
| `X_signal` | `np.ndarray` | Input points for the signal region |
| `t_signal` | `np.ndarray` | Observed signal counts |
| `bounds_theta` | list of tuples, optional | Bounds for PoLoN hyperparameters. Default: `[(0.0001, 2), (0.01, 50)]` |
| `theta_init` | `np.ndarray`, optional | Initial guess for PoLoN hyperparameters. Default: `[0.05, 20.0]` |
| `bounds_signal` | list of tuples, optional | Bounds for the localized signal parameters (scaled). Default: `[(0.01, 10), (μ_min/0.1, μ_max/0.1), (1.0, 50.0)]` |
| `signal_params0_scaled` | list or `np.ndarray`, optional | Initial guess for the scaled signal parameters. Default: `[1.0, mean(X_signal)/0.1, 0.01]` |
| `X_input` | `np.ndarray`, optional | Input points where predictions are evaluated. Default: 100 evenly spaced points across all data. |


##### **Outputs**

Returns a dictionary containing:

- `theta_opt`: optimized PoLoN hyperparameters  
- `A_opt`: optimized signal amplitude  
- `mu_opt`: optimized signal mean (location)  
- `sigma_opt`: optimized signal width (spread)  
- `poisson_mean_background`: background contribution (PoLoN)  
- `poisson_mean_total`: total PoLoN prediction (signal + background)  
- `mu_values`: latent PoLoN mean field  
- `std_values`: latent PoLoN standard deviation field  
- `signal_component`: fitted localized signal contribution  
- `theta_success`, `signal_success`: optimization success flags  


##### **Visualization Output**

When executed, this function automatically generates **two plots**:

1. **Total PoLoN Prediction vs Observed Data**  
   - Blue: total PoLoN prediction (signal + background)  
   - Green dashed: background-only component  
   - Orange / gray dots: observed data points (signal and background)  
   - Shaded blue region: uncertainty band from PoLoN fluctuations  

2. **Signal and Background Decomposition**  
   - Red: localized signal component  
   - Green dashed: background component  
   - Blue: total PoLoN model output  

These visualizations clearly demonstrate how the PoLoN process captures both the stochastic background and the localized signal within a single probabilistic framework.


##### **Example: Using `predict_signal_background_with_plot` with a signal window**

```python
from helper import *  # adjust import as needed

# Load unbinned data from repo
unbinned_data = np.load("data/unbinned_diphoton_mass.npy")  # relative path in repo

# Histogram setup
xmin, xmax, step_size = 99.5, 160.5, 1.0
bin_edges = np.arange(xmin, xmax + step_size, step_size)
bin_centres = np.arange(xmin + step_size/2, xmax + step_size/2, step_size)
data_counts, _ = np.histogram(unbinned_data, bins=bin_edges)

# --- Define a signal window (user chooses) ---
signal_window = (bin_centres >= 120) & (bin_centres <= 140)

# Split data manually into signal and background regions
X_signal = bin_centres[signal_window]/1000
t_signal = data_counts[signal_window]

X_bg = bin_centres[~signal_window]/1000
t_bg = data_counts[~signal_window]

# --- Run PoLoN prediction with background and signal inputs ---
results = predict_signal_background_with_plot(X_bg, t_bg, X_signal, t_signal)

print("Optimized PoLoN hyperparameters:", results["theta_opt"])
print("Optimized Gaussian signal (A, μ, σ):", results["A_opt"], results["mu_opt"], results["sigma_opt"])

```
> **Note 1 :** This example demonstrates one way to define a **signal window** for separating background and signal regions in the data. The function itself does not automatically separate background and signal; you need to provide `X_bg`, `t_bg`, `X_signal`, and `t_signal` based on your chosen window.




> **Note 2 :**  
> This function can be computationally expensive, especially when exploring multiple signal strengths or realizations.  
> For large-scale experiments, **parallelization (e.g., via Amarel job submission or HPC clusters)** is recommended.  
> The function itself is defined in `helper.py` for modular reuse and does not appear directly in the Jupyter notebook.
---
### Tips & Best Practices

- **Choosing the Right Function:**  
  - Use `polon_predict_and_plot()` for smooth, background-dominated count data or general PoLoN predictions.  
  - Use `predict_signal_background_with_plot()` when there is a **localized signal** whose approximate position is known. This function provides additional information about the signal’s properties (amplitude, mean location, and width).

- **Defining a Signal Window:**  
  - The signal window is **user-defined**. You need to provide `X_bg`, `t_bg` for background and `X_signal`, `t_signal` for the signal region.  
  - This approach allows you to explore different signal regions without modifying the function itself.

- **Using Your Own Data:**  
  - Place datasets in the `data/` folder.  
  - Update file paths in the notebook or scripts accordingly. Supported formats include `.csv` and `.npy`.  
  - Ensure that the data is formatted consistently with the examples.

- **Performance Considerations:**  
  - The functions, especially `predict_signal_background_with_plot()`, can be computationally intensive for large datasets or multiple signal realizations.  
  - For heavy computations, consider **parallel execution**, HPC clusters, or job submission systems like Amarel.

- **Modularity and Reusability:**  
  - All PoLoN functions are defined in `helper.py`.  
  - Importing this module is sufficient to access the core functionality; the notebook itself does not need to be modified for reusing the code.

- **Visualization:**  
  - Both functions automatically generate plots to visualize predictions, uncertainties, and signal-background decomposition (if applicable).  
  - Use the plots to validate your signal window choice and assess model quality.
    





