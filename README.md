# Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data

### Overview
Modeling datasets of integer counts is crucial in physics and other scientific disciplines, where measurements involve discrete, non-negative quantities. Traditional approaches such as Poisson regression often struggle to capture complex, non-linear relationships.

The **Poisson Log-Normal (PoLoN) process** models the Poisson rate as the exponential of a latent Gaussian process and propagates the uncertainty in that latent rate through the Poisson likelihood. This gives a closed-form predictive mean for integer count data, with the most probable count and the 95% interval obtained by Monte Carlo sampling.

This repository provides:
- Core implementations of PoLoN for predicting expected counts and most probable outcomes.
- Signal + background modeling (PoLoN-SB) with either a user-specified signal region or a sliding-window search when the signal location is unknown.
- Real-data examples (Higgs diphoton mass spectrum, bike rental demand) and small synthetic examples.

---

### Key Features
- Gaussian Process prior on the latent log-rate of a Poisson likelihood
- Non-parametric prediction of integer counts: expected counts, most probable counts and 95% intervals
- Handles discrete, non-negative data
- Replicate-aware: repeated observations at the same input are aggregated, so the covariance matrix is M × M over the M unique inputs rather than over every observation
- Signal + background decomposition: a localized Gaussian signal on top of a smooth GP background
- Sliding-window search for a localized signal of unknown position

---

### Repository Structure
```
PoLoN-Process/
│
├── data/
│   ├── hour.csv
│   └── unbinned_diphoton_mass.npy
├── scripts/
│   └── unknown_location_signal_search.py
├── LICENSE
├── README.md
├── helpers.py
└── requirements.txt
```

- `helpers.py` contains all reusable functions (see [Functions in `helpers.py`](#functions-in-helperspy)).
- `scripts/unknown_location_signal_search.py` runs the sliding-window search for a signal of unknown location.

---

### Setup
Python 3.8 or higher is supported (tested on 3.8 and 3.12). Install the dependencies with:

```bash
pip install -r requirements.txt
```

The dependencies are numpy, scipy, matplotlib and pandas. matplotlib is only needed for plotting; pandas is used by the bike-rental example and the search script.

---

### Quick Start

The following example fits a small PoLoN model and a small signal + background model to synthetic data, shows the figures, and prints the fitted values next to the true ones. Save it as a `.py` file in the repository root (or paste it into a Python session started there) so that `helpers.py` can be imported.

```python
import numpy as np
from helpers import *

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. PoLoN
x = np.linspace(0.1, 0.16, 10)
alpha = 40 * np.exp(-30.6 * (x - 0.1))              
X_train = np.repeat(x, 3)                            # 3 replicates at each input
t_train = rng.poisson(np.repeat(alpha, 3))

results = polon_predict_and_plot(X_train, t_train,
                                 theta_init=np.array([0.05, 20.0]),
                                 bounds=[(0.0001, 3), (0.01, 50)],
                                 n_samples=1000)
print("\nPoLoN: predicted mean (first 5 grid points):", results["poisson_mean_output"][:5].round(2))
print("PoLoN: most probable counts (first 5):      ", results["most_probable_output"][:5])

# ------------------------------------------------- 2. signal + background
c0, lam, q, S, u = np.exp(11.7), 30.6, 0.135, 300, 0.004      # true signal: S=300, q=0.135, u=0.004
xu = np.linspace(0.1, 0.16, 24)
rate = c0 * np.exp(-lam * xu) + S * np.exp(-(xu - q) ** 2 / (2 * u ** 2))
X_all = np.repeat(xu, 3)
t_all = rng.poisson(np.repeat(rate, 3))
in_window = np.repeat(np.abs(xu - q) <= 2 * u, 3)             # signal window: q +- 2u

res = predict_signal_background_with_plot(X_all[~in_window], t_all[~in_window],
                                          X_all[in_window], t_all[in_window],
                                          X_input=xu,
                                          bounds_theta=[(0.0001, 3), (0.01, 50)],
                                          n_samples=1000)
print(f"\nSignal fit:  A = {res['A_opt']:.1f} (true {S}),  q = {res['mu_opt']:.4f} (true {q}),"
      f"  u = {res['sigma_opt']:.5f} (true {u})")
print("Optimisers converged:", res["theta_success"], res["signal_success"])
print("Most probable counts (first 5 grid points):", res["most_probable_output"][:5])
```

---

### Data Directory and Example Datasets

Example datasets are placed in the `data/` folder. You can include `.csv` or `.npy` files. The current example datasets are:

| File Name                        | Type   | Description |
|----------------------------------|--------|-------------|
| `hour.csv`                        | CSV    | Hourly bike rental counts (real-world example) |
| `unbinned_diphoton_mass.npy`      | NPY    | Unbinned diphoton mass data used for the signal-background PoLoN example (Higgs search) |

**Usage Example:**

```python
import numpy as np
import pandas as pd

# Load unbinned diphoton mass dataset
data = np.load("data/unbinned_diphoton_mass.npy")

# Load bike rental dataset
bike_data = pd.read_csv("data/hour.csv")
```

---

### Functions in `helpers.py`

The high-level functions are the main entry points:

| Function | Purpose |
|---|---|
| `polon_predict_and_plot()` | PoLoN prediction: fit, predict and plot |
| `predict_signal_background_with_plot()` | Signal + background model for a specified signal window: fit, predict and plot |

Each has a predict-only counterpart that returns a results dictionary without plotting, and a plot-only function that draws from that dictionary without refitting:

| Predict only | Plot only |
|---|---|
| `polon_predict()` | `plot_polon_prediction()` |
| `predict_signal_background()` | `plot_signal_background()` |

The lower-level functions implement the individual components:

| Group | Functions |
|---|---|
| Kernel and covariance | `rbf_kernel`, `compute_cov_matrix`, `compute_k_vec` |
| Repeated observations | `aggregate_replicates` |
| Laplace approximation | `compute_residual`, `compute_gradient`, `compute_hessian`, `newton_optimization`, `log_det_cholesky` |
| Hyperparameters | `neg_log_likelihood`, `optimise_hyperparameters` |
| Prediction | `predict`, `predict_many` |
| Signal + background likelihood | `gaussian_bump`, `background_loglik`, `signal_loglik`, `total_loglik`, `neg_total_loglik_scaled` |

**How repeated observations are handled.** If several observations share the same input, `aggregate_replicates(X_all, t_all)` collapses them into the unique inputs `X` (M of them), the total count `T` and the number of replicates `Nv` at each. The Laplace approximation and the covariance matrix then work with M unique inputs, however many replicates there are.

**Prediction at a point.** `predict(X_new, X, T, Nv, theta)` returns the mean and variance of the latent log-rate and which case was used: the Gaussian-process conditional for a new input (`"A"`), or the Laplace marginal for an input that was in the training data (`"B"`). `predict_many` applies it to many inputs. The PoLoN predictive mean is `exp(mu + std**2 / 2)`.

---

#### 1. PoLoN Predictive Modeling

The basic PoLoN predictive workflow is implemented in `polon_predict_and_plot()`. Use it for integer count data without separating signal and background components.

##### **Function Overview**

- Trains the PoLoN model on the training data `(X_train, t_train)`; repeated inputs are aggregated automatically.
- Optimizes the kernel hyperparameters by maximizing the Laplace marginal likelihood.
- Predicts the **expected counts** (Poisson mean) and **most probable counts** (Poisson mode) for new input points.
- Computes **Monte Carlo-based 95% intervals** for the predicted counts.
- Generates plots of the predictive mean, the intervals and the most probable outputs.

##### **Inputs**

| Parameter      | Type       | Description |
|----------------|------------|-------------|
| `X_train`      | `np.ndarray` | Training inputs (1D, or 2D with one row per observation) |
| `t_train`      | `np.ndarray` | Training counts, one per observation |
| `X_input`      | `np.ndarray`, optional | Points at which to predict. Default: 100 evenly spaced points across the training range (1D inputs only) |
| `theta_init`   | `np.ndarray`, optional | Initial guess for the kernel hyperparameters. Default: `[0.01, 20.0]` |
| `bounds`       | list of tuples, optional | Bounds for the hyperparameters during optimization. Default: `[(0.0001, 30), (0.01, 30)]` |
| `n_samples`    | int, optional | Number of Monte Carlo samples. Default: 5000 |
| `plot`         | bool, optional | Draw the figure (1D inputs only). Default: `True` |
| `theta`        | tuple, optional | Kernel hyperparameters `(sigma, gamma)`. If given, the hyperparameter optimization is skipped |

##### **Outputs**

A dictionary containing:

- `X_input`: input points where predictions were made  
- `mu_values`: predictive mean of the latent log-rate  
- `std_values`: predictive standard deviation of the latent log-rate  
- `poisson_mean_output`: predictive mean of the Poisson counts  
- `lower_bounds`, `upper_bounds`: Monte Carlo-based 95% interval  
- `most_probable_output`: most probable Poisson count (mode)  
- `theta_opt`: kernel hyperparameters used (optimized, or the ones supplied)  
- `theta_success`: optimizer success flag (`None` if `theta` was supplied)  
- `X_train`, `t_train`: the training data, so that the results can be plotted on their own  

##### **Example**

```python
import numpy as np
from helpers import *

# Synthetic training data: a rate falling from 40 to about 6, with 3 replicates per input
rng = np.random.default_rng(0)
x = np.linspace(0.1, 0.16, 10)
rate = 40 * np.exp(-30.6 * (x - 0.1))
X_train = np.repeat(x, 3)
t_train = rng.poisson(np.repeat(rate, 3))

# Fit, predict and plot
results = polon_predict_and_plot(X_train, t_train,
                                 theta_init=np.array([0.05, 20.0]),
                                 bounds=[(0.0001, 3), (0.01, 50)],
                                 n_samples=1000)

print("Predicted Poisson mean (first 5):", results["poisson_mean_output"][:5].round(2))
print("Most probable counts (first 5):  ", results["most_probable_output"][:5])
print("95% interval at the first point: ", results["lower_bounds"][0], "to", results["upper_bounds"][0])
```

The predicted mean should fall from about 39 to about 7 across the range.

##### **Redrawing and reusing a fit**

The results dictionary contains everything needed to plot, so a figure can be redrawn without refitting. Passing `theta` skips the hyperparameter optimization.

```python
plot_polon_prediction(results)                                            # redraw without refitting
results2 = polon_predict(X_train, t_train, theta=results["theta_opt"])    # reuse the hyperparameters, no plot
```

---

#### 2. Signal + Background Modeling

The signal-background extension is implemented in `predict_signal_background_with_plot()`. Use it when the signal region is specified in advance: you provide the background and signal regions separately through `X_bg`, `t_bg`, `X_signal` and `t_signal`. The function does not determine the signal location itself (see [Unknown Signal Location Search](#3-unknown-signal-location-search) for that).

This approach is relevant in **physics applications** (e.g. Higgs boson searches), where the observed data are composed of a smooth stochastic background and a localized signal peak.

##### **Function Overview**

- Optimizes the **PoLoN hyperparameters** using the background region only.
- Fits a **localized Gaussian signal** (amplitude, location, width) on top of the fitted background.
- Predicts the full intensity (signal + background) across the input range.
- Computes the most probable count and a 95% interval of the total prediction by Monte Carlo sampling.
- Generates two plots (see below).

##### **Inputs**

| Parameter | Type | Description |
|------------|------|-------------|
| `X_bg` | `np.ndarray` | Input points for the background region |
| `t_bg` | `np.ndarray` | Observed background counts |
| `X_signal` | `np.ndarray` | Input points for the signal region |
| `t_signal` | `np.ndarray` | Observed signal-region counts |
| `bounds_theta` | list of tuples, optional | Bounds for the PoLoN hyperparameters. Default: `[(0.0001, 2), (0.01, 50)]` |
| `theta_init` | `np.ndarray`, optional | Initial guess for the PoLoN hyperparameters. Default: `[0.05, 20.0]` |
| `bounds_signal` | list of tuples, optional | Bounds for the signal parameters (scaled, see the note below). Default: `[(0.01, 10), (μ_min/0.1, μ_max/0.1), (1.0, 50.0)]` |
| `signal_params0_scaled` | list or `np.ndarray`, optional | Initial guess for the scaled signal parameters. Default: `[1.0, mean(X_signal)/0.1, 4.0]` |
| `X_input` | `np.ndarray`, optional | Points where predictions are evaluated. Default: 100 evenly spaced points across all data |
| `n_samples` | int, optional | Number of Monte Carlo samples. Default: 5000 |
| `plot` | bool, optional | Draw the figure (1D inputs only). Default: `True` |
| `theta` | tuple, optional | Kernel hyperparameters `(sigma, gamma)`; skips the background optimization |
| `signal_params` | tuple, optional | Signal parameters `(A, mu, sigma)` in input units; skips the signal optimization |

##### **Outputs**

A dictionary containing:

- `theta_opt`: PoLoN hyperparameters used  
- `A_opt`: signal amplitude  
- `mu_opt`: signal mean (location)  
- `sigma_opt`: signal width (spread)  
- `poisson_mean_background`: background contribution (PoLoN)  
- `poisson_mean_total`: total prediction (signal + background)  
- `mu_values`, `std_values`: latent PoLoN mean and standard deviation of the background  
- `signal_component`: fitted localized signal contribution  
- `lower_bounds`, `upper_bounds`: Monte Carlo-based 95% interval of the total prediction  
- `most_probable_output`: most probable count of the total prediction  
- `theta_success`, `signal_success`: optimization success flags (`None` if the corresponding parameters were supplied)  
- `X_input`, `X_bg`, `t_bg`, `X_signal`, `t_signal`: the points and data used, so that the results can be plotted on their own  

##### **Visualization Output**

When executed, the function generates **two plots**; `plot_signal_background(results)` redraws them from the results dictionary:

1. **Mean prediction**  
   - Orange dots: observed data points (signal and background regions)  
   - Blue: total prediction (signal + background)  
   - Green dashed: background-only component  
   - Shaded blue region: 95% interval of the predicted counts  

2. **Most probable output**  
   - Orange dots: observed data points  
   - Blue dots: most probable count at each prediction point  

##### **Example: synthetic data with a known signal**

```python
import numpy as np
from helpers import *

# Exponential background plus a Gaussian signal: strength S=300, location q=0.135, width u=0.004
rng = np.random.default_rng(1)
c0, lam, q, S, u = np.exp(11.7), 30.6, 0.135, 300, 0.004
x = np.linspace(0.1, 0.16, 24)
rate = c0 * np.exp(-lam * x) + S * np.exp(-(x - q) ** 2 / (2 * u ** 2))
X_all = np.repeat(x, 3)                                   # 3 replicates per input
t_all = rng.poisson(np.repeat(rate, 3))

# Define the signal window (q +- 2u) and split the data into the two regions
in_window = np.repeat(np.abs(x - q) <= 2 * u, 3)

results = predict_signal_background_with_plot(
    X_all[~in_window], t_all[~in_window],       # background region
    X_all[in_window],  t_all[in_window],        # signal region
    bounds_theta=[(0.0001, 3), (0.01, 50)],
    n_samples=1000)

print(f"Fitted signal: A = {results['A_opt']:.1f}, location = {results['mu_opt']:.4f}, width = {results['sigma_opt']:.5f}")
print(f"True signal:   A = {S},   location = {q},    width = {u}")
```

##### **Example: Higgs diphoton spectrum with a signal window**

```python
import numpy as np
from helpers import *

# Load unbinned data from the repo
unbinned_data = np.load("data/unbinned_diphoton_mass.npy")

# Histogram setup
xmin, xmax, step_size = 99.5, 160.5, 1.0
bin_edges = np.arange(xmin, xmax + step_size, step_size)
bin_centres = np.arange(xmin + step_size/2, xmax + step_size/2, step_size)
data_counts, _ = np.histogram(unbinned_data, bins=bin_edges)

# --- Define a signal window (user chooses) ---
signal_window = (bin_centres >= 120) & (bin_centres <= 140)

# Split data manually into signal and background regions (inputs in units of TeV, i.e. GeV / 1000)
X_signal = bin_centres[signal_window] / 1000
t_signal = data_counts[signal_window]

X_bg = bin_centres[~signal_window] / 1000
t_bg = data_counts[~signal_window]

# --- Run PoLoN prediction with background and signal inputs ---
results = predict_signal_background_with_plot(X_bg, t_bg, X_signal, t_signal)

print("Optimized PoLoN hyperparameters:", results["theta_opt"])
print("Optimized Gaussian signal (A, μ, σ):", results["A_opt"], results["mu_opt"], results["sigma_opt"])
```

> **Note 1:** This example demonstrates one way to define a **signal window** for separating background and signal regions in the data. The function does not separate background and signal automatically; you need to provide `X_bg`, `t_bg`, `X_signal` and `t_signal` based on your chosen window.

> **Note 2:** This approach relies on the assumption that the signal is weak relative to the background. We plan to investigate the strong-signal regime in future work.

> **Note 3:** This function can be computationally expensive, especially when exploring multiple signal strengths or realizations. For large-scale experiments, **parallelization (e.g. via cluster job submission)** is recommended.

> **Note 4 (units):** the signal parameters are optimized in scaled variables: amplitude = 100 × `a`, location = 0.1 × `m`, width = 0.001 × `s`. The defaults assume inputs of order 0.1 (as above, where GeV is divided by 1000) and signal widths between 0.001 and 0.05 in input units. For data on a different scale, rescale the inputs or adjust `bounds_signal` and `signal_params0_scaled`.

---

#### 3. Unknown Signal Location Search

In many applications the location of a localized signal is not known in advance. The repository provides a sliding-window search that looks for candidate signal regions in the count data.

The method divides the input range into overlapping candidate windows. For each window, the points inside it are treated as a possible signal region, while the remaining points are used to estimate the PoLoN background. A Gaussian signal is then fitted within the candidate window, and the fitted parameters and likelihoods are recorded.

The script `scripts/unknown_location_signal_search.py` generates a synthetic Poisson count dataset containing a localized Gaussian signal and runs the search on it. The true signal strength and location are used only to generate the data; they are not given to the search.

##### Running the search

Run it from the repository root, as a module, so that `helpers.py` is found:

```bash
python -m scripts.unknown_location_signal_search --signal_strength 300 --n_instances_per_X 3 --seed 42 --window_size 10 --signal_location 0.135 --output_file scan_results.csv
```

> **Note:** `python scripts/unknown_location_signal_search.py` fails with `No module named 'helpers'` when run from the repository root, because Python only searches the script's own folder. Use the command above, or set `PYTHONPATH=.` first.

| Option | Default | Description |
|---|---|---|
| `--signal_strength` | 50 | Amplitude of the injected Gaussian signal |
| `--n_instances_per_X` | 20 | Number of replicate observations at each of the 40 inputs |
| `--seed` | 42 | Random seed for the synthetic data |
| `--window_size` | 10 | Number of unique inputs in each candidate window |
| `--signal_location` | 0.135 | Location of the injected signal |
| `--output_file` | `scan_results.csv` | Output file (overwritten if it exists) |

The search is slow with the solver in `helpers.py`: roughly a minute per window for 40 inputs, so about 30 minutes for the default window size of 10 (31 windows). One row is written to the output file as soon as each window is finished.

##### Output

One row per candidate window:

| Column | Description |
|---|---|
| `window_index`, `window_center`, `x_min`, `x_max` | Position of the candidate window |
| `ll_bg` | Log-likelihood of the points inside the window under the background-only model |
| `ll_signal` | Log-likelihood of the same points under background + fitted signal |
| `TS` | Test statistic, `2 * (ll_signal - ll_bg)` |
| `A_hat`, `mu_hat`, `sigma_hat` | Fitted signal amplitude, location and width |
| `theta0`, `theta1` | PoLoN background hyperparameters |
| `N_signal`, `N_bg` | Number of observations inside / outside the window |
| `ll_tot` | Log-likelihood of all points under the fitted signal + background model |

A large `TS` indicates evidence for a signal in that window. It is a likelihood-ratio statistic for comparing windows; no significance calibration is provided. The background length scale is optimized with a sigmoid regularization term (weight `penalty`, 5000 by default, set in `run_scan()`) that favors a smooth background, so that the background does not absorb the signal.

##### Analysing the output

```python
import numpy as np
import pandas as pd

df = pd.read_csv("scan_results.csv")

# Window that explains the data best
best = df.loc[df["ll_tot"].idxmax()]
print(best[["window_center", "A_hat", "mu_hat", "sigma_hat", "TS"]])

# Likelihood-weighted estimate across all windows
w = np.exp(df["ll_tot"] - df["ll_tot"].max())
for col in ["A_hat", "mu_hat", "sigma_hat"]:
    print(col, (w * df[col]).sum() / w.sum())
```

The weighting gives more weight to windows with larger `ll_tot`, which gives a single estimate of the signal amplitude, location and width.

To search your own data, replace the call to `generate_data()` in `run_scan()` with your data: `X_raw` as an `(N, 1)` array of inputs, `t_raw` as an `(N,)` array of counts and `X_unique = np.unique(X_raw[:, 0])`.

---

#### 4. Bike Rental Example

The bike-rental data (`data/hour.csv`) have repeated observations at the same (hour, weekday) input, which `aggregate_replicates` handles. The example fits the odd hours of the first three weeks and predicts the even hours:

```python
import numpy as np
import pandas as pd
from helpers import *

df = pd.read_csv("data/hour.csv").iloc[:479]
train, test = df[df["hr"] % 2 != 0], df[df["hr"] % 2 == 0]
X_train = train[["hr", "weekday"]].to_numpy(float)
y_train = train["cnt"].to_numpy(float)
X_test = test[["hr", "weekday"]].to_numpy(float)
y_test = test["cnt"].to_numpy(float)

# Collapse repeated (hour, weekday) inputs: 82 unique inputs
X, T, Nv = aggregate_replicates(X_train, y_train)

# Kernel hyperparameters from a previous fit of these data.
# To refit, use optimise_hyperparameters(X, T, Nv, theta_init) (slow).
theta = (2.1939478, 7.19102992)
C = compute_cov_matrix(X, *theta, regularization=1e-6)
lam_star, _ = newton_optimization(T, Nv, C)

# Predict the held-out even hours
mu, std = predict_many(X, T, Nv, theta, X_test, lam_star=lam_star, c_reg=1e-6)
y_pred = np.exp(mu + std ** 2 / 2)

r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"Unique inputs: {X.shape[0]},  test R^2 = {r2:.3f}")
```

The test R² should be about 0.84.

---

### Tips & Best Practices

- Use `polon_predict_and_plot()` when the primary goal is PoLoN-based count-data prediction.
- Use `predict_signal_background_with_plot()` when a localized signal is present and the signal region can be specified in advance.
- Use the search script in `scripts/` when the signal location is not known in advance.
- Use the predict-only functions (`polon_predict`, `predict_signal_background`) in scripts and batch jobs, and the plot-only functions to draw or restyle figures afterwards without refitting.
- Pass `theta` (and `signal_params`) to reuse hyperparameters you have already fitted. This skips the slowest step.
- For large datasets or repeated analyses, consider the computational cost of the Laplace approximation and hyperparameter optimization.

- **Using your own data:**  
  - Place datasets in the `data/` folder. Supported formats include `.csv` and `.npy`.  
  - Provide one observation per entry: `X` with one row per observation (repeated rows are fine) and `t` with the matching counts.  
  - The signal fit assumes inputs of order 0.1 (see Note 4 above).  

- **Performance considerations:**  
  - The Laplace solver `newton_optimization` takes conservative damped steps, which makes every likelihood evaluation slow. `predict_signal_background_with_plot()` and the search script can run for many minutes.  
  - For heavy computations, consider **parallel execution**, an HPC cluster, or a job submission system.  

- **Reproducibility:**  
  - Results are deterministic for a given machine, data and seed. On a nearly flat likelihood the optimizer can stop at slightly different hyperparameters on a different machine or library version, which can shift weak-window `TS` values by several percent; the predictions change very little.  

- **Visualization:**  
  - The plotting functions handle 1D inputs only.  
  - Use the plots to validate your signal window choice and assess model quality.
