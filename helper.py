# helpers.py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, poisson


def compute_M(t, lambda_vector):
    M = t - np.exp(lambda_vector)
    return M

def rbf_kernel_extended(x, y, sigma=1.0, gamma=1.0):
    dist_squared = np.sum((x - y) ** 2)
    return gamma * np.exp(-dist_squared / (2 * sigma ** 2))

def compute_cov_matrix(X, sigma=1.0, gamma=1.0, regularization=1e-5):
    num_points = X.shape[0]
    covariance_matrix = np.zeros((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            covariance_matrix[row, col] = rbf_kernel_extended(X[row], X[col], sigma, gamma)
    np.fill_diagonal(covariance_matrix, np.diagonal(covariance_matrix) + regularization)
    return covariance_matrix

def compute_K(X, X_new, sigma=1.0, gamma=1.0):
    n = X.shape[0]
    row_matrix = np.zeros(n)
    for i in range(n):
        row_matrix[i] = rbf_kernel_extended(X[i], X_new, sigma, gamma)
    return row_matrix

# Kernel function (using linear kernel with two parameters)
def linear_kernel(x, y, theta1=1.0, theta2=0.0):
    return theta1 * np.dot(x, y) + theta2

# Compute covariance matrix with two parameters
def compute_cov_matrix_lin(X, theta1=1.0, theta2=0.0, regularization=1e-4):
    n = X.shape[0]
    Cn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Cn[i, j] = linear_kernel(X[i], X[j], theta1, theta2)
    
    # Add small regularization term to the diagonal
    np.fill_diagonal(Cn, np.diagonal(Cn) + regularization)
    
    return Cn
    
def compute_K_lin(X, X_new, theta1=1.0, theta2 = 0.0):
    n = X.shape[0]
    row_matrix = np.zeros(n)
    for i in range(n):
        row_matrix[i] = linear_kernel(X[i], X_new, theta1, theta2)  # Call the RBF kernel with two parameters
    return row_matrix

def compute_gradient(t, lambda_vector, Cn):
    M = t - np.exp(lambda_vector)
    Cn_inv = np.linalg.inv(Cn)
    grad_f = M - Cn_inv @ lambda_vector
    return grad_f

def compute_hessian(Cn, lambda_vector):
    Cn_inv = np.linalg.inv(Cn)
    Wn = np.diag(np.exp(lambda_vector))
    H = Cn_inv + Wn
    return H

def log_det_cholesky(matrix):
    L = np.linalg.cholesky(matrix)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return log_det

def newton_optimization(t, Cn, max_iter=20000, tol=1e-6):
    lambda_init = np.ones(Cn.shape[0])
    for _ in range(max_iter):
        grad_f = compute_gradient(t, lambda_init, Cn)
        H = compute_hessian(Cn, lambda_init)
        change = 0.001 * np.linalg.inv(H).dot(grad_f)
        lambda_init += change
        if np.max(np.abs(change)) < tol:
            break
    return lambda_init, compute_hessian(Cn, lambda_init)

def neg_log_likelihood_function(theta, X, t):
    Cn = compute_cov_matrix(X, *theta)
    lambda_n_star, _ = newton_optimization(t, Cn)
    H = compute_hessian(Cn, lambda_n_star)
    log_likelihood_value = (
        t.T @ lambda_n_star
        - np.sum(np.exp(lambda_n_star))
        - 0.5 * log_det_cholesky(H)
        + multivariate_normal.logpdf(lambda_n_star, mean=np.zeros(X.shape[0]), cov=Cn)
    )
    return -log_likelihood_value

def gaussian_bump(X, A, mu, sigma):
    return A * np.exp(-((X - mu)**2) / (2 * sigma**2))

def signal_loglik_and_hessian(X_bg, t_signal, X_signal, lambda_n_star, theta, A, mu, sigma):
    Cn = compute_cov_matrix(X_bg, *theta)
    invCn = np.linalg.inv(Cn)
    H_gp = compute_hessian(Cn, lambda_n_star)
    invH_gp = np.linalg.inv(H_gp)
    poisson_rates = []
    H_signal = np.zeros_like(Cn)
    for X_new, t in zip(X_signal, t_signal):
        K = compute_K(X_bg, X_new, *theta)
        L = K.T @ invCn
        mean_pred = L @ lambda_n_star
        var_pred = abs(L @ invH_gp @ L.T + 1e-5)
        mean_log = mean_pred + var_pred / 2
        g = gaussian_bump(X_new, A, mu, sigma)
        lam = np.exp(mean_log) + g
        poisson_rates.append(lam)
        coeff = (t * g / (lam**2) - 1) * np.exp(mean_log)
        H_signal += coeff * np.outer(L, L)
    loglik_signal = np.sum(poisson.logpmf(t_signal, poisson_rates))
    return loglik_signal, H_signal

def background_loglik_and_hessian(Cn, X_bg, t_bg, lambda_n_star, A, mu, sigma):
    Cn_inv = np.linalg.inv(Cn)
    G = gaussian_bump(X_bg, A, mu, sigma)
    exp_lambda = np.exp(lambda_n_star)
    denom = 1.0 + G / exp_lambda
    W_diag = exp_lambda - (t_bg * G) / (exp_lambda * denom**2)
    Wn = np.diag(W_diag)
    H_background = Cn_inv + Wn
    rate = exp_lambda + G
    loglik_background = np.sum(poisson.logpmf(t_bg, rate))
    return loglik_background, H_background

def total_loglik_and_hessian(X_bg, t_bg, X_signal, t_signal, theta, A, mu, sigma):
    Cn = compute_cov_matrix(X_bg, *theta)
    lambda_n_star, _ = newton_optimization(t_bg, Cn)
    loglik_bg, H_bg = background_loglik_and_hessian(Cn, X_bg, t_bg, lambda_n_star, A, mu, sigma)
    loglik_signal, H_signal = signal_loglik_and_hessian(X_bg, t_signal, X_signal, lambda_n_star, theta, A, mu, sigma)
    total_loglik = loglik_bg + loglik_signal
    H_total = H_bg - H_signal
    det_term = -0.5 * log_det_cholesky(H_total)
    tot_log_lik = total_loglik + det_term
    return loglik_bg, loglik_signal, det_term, tot_log_lik

def neg_total_loglik_scaled(params_scaled, X_bg, t_bg, X_signal, t_signal, theta):
    A_scaled, mu_scaled, sigma_scaled = params_scaled
    A = A_scaled * 100
    mu = mu_scaled * 0.1
    sigma = sigma_scaled * 0.001
    _, _, _, tot_log_lik = total_loglik_and_hessian(X_bg, t_bg, X_signal, t_signal, theta, A, mu, sigma)
    return -tot_log_lik

def polon_predict_and_plot(X_train, t_train, X_input=None, theta_init=None, bounds=None, n_samples=5000):
    """
    Perform PoLoN prediction and plot mean and most probable Poisson outputs.

    Parameters
    ----------
    X_train : np.ndarray
        Training input values (1D or 2D array of features)
    t_train : np.ndarray
        Training output counts
    X_input : np.ndarray, optional
        Input values for predictions. Default: 100 points in training range
    theta_init : np.ndarray, optional
        Initial guess for kernel hyperparameters. Default: [0.01, 20.0]
    bounds : list of tuple, optional
        Bounds for hyperparameters in minimization. Default: [(0.0001, 30), (0.01, 30)]
    n_samples : int, optional
        Number of Monte Carlo samples for most probable Poisson calculation

    Returns
    -------
    dict
        Dictionary containing predictive mean, std, Monte Carlo confidence intervals,
        and most probable Poisson outputs.
    """
    
    X = np.array(X_train, dtype=np.float64)
    t = np.array(t_train, dtype=np.float64)

    # Default theta and bounds if not provided
    if theta_init is None:
        theta_init = np.array([0.01, 20.0], dtype=np.float64)
    if bounds is None:
        bounds = [(0.0001, 30), (0.01, 30)]

    # Step 1: Optimize hyperparameters
    result = minimize(
        fun=neg_log_likelihood_function,
        x0=theta_init,
        args=(X, t),
        method='L-BFGS-B',
        bounds=bounds
    )
    theta_opt = result.x
    print(f"Optimized theta: {theta_opt}")
    print(f"Log-likelihood at optimum: {-result.fun:.4f}")

    # Step 2: Compute Cn with optimized theta
    Cn = compute_cov_matrix(X, *theta_opt)

    # Step 3: Solve for lambda_predicted using Newton optimization
    lambda_init = np.ones(X.shape[0])
    lambda_predicted, _ = newton_optimization(t, Cn, max_iter=25000, tol=1e-5)

    # Step 4: Define prediction input points
    if X_input is None:
        X_input = np.linspace(np.min(X), np.max(X), 100)

    mu_values = []
    std_values = []

    # Step 5: Predict mean and variance at new points
    for X_new in X_input:
        K = compute_K(X, X_new, *theta_opt)
        mean_lambda = K.T @ np.linalg.inv(Cn) @ lambda_predicted
        H = compute_hessian(Cn, lambda_predicted)
        var_lambda = abs(
            rbf_kernel_extended(X_new, X_new, *theta_opt) + 1e-5
            - K.T @ np.linalg.inv(Cn) @ K
            + K.T @ np.linalg.inv(Cn) @ np.linalg.inv(H) @ np.linalg.inv(Cn) @ K
        )
        mu_values.append(mean_lambda.item())
        std_values.append(np.sqrt(var_lambda))

    mu_values = np.array(mu_values)
    std_values = np.array(std_values)

    # Step 6: Compute Poisson predictive mean
    poisson_mean_output = np.exp(mu_values + std_values**2 / 2)

    # Step 7: Monte Carlo for most probable Poisson outputs and confidence intervals
    most_probable_output = []
    lower_bounds = []
    upper_bounds = []

    k_range = np.arange(0, int(np.max(poisson_mean_output)*2)+10)

    for mu, sigma in zip(mu_values, std_values):
        lambda_samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
        monte_carlo_probs = [np.mean(poisson.pmf(k, lambda_samples)) for k in k_range]
        monte_carlo_probs = np.array(monte_carlo_probs)
        cdf = np.cumsum(monte_carlo_probs)
        
        # Most probable value
        k_argmax = k_range[np.argmax(monte_carlo_probs)]
        most_probable_output.append(k_argmax)
        
        # 95% confidence interval
        k_lower = k_range[np.searchsorted(cdf, 0.025)]
        k_upper = k_range[np.searchsorted(cdf, 0.975)]
        lower_bounds.append(k_lower)
        upper_bounds.append(k_upper)

    most_probable_output = np.array(most_probable_output)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Step 8: Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Predictive mean plot
    axes[0].scatter(X, t, label="Training Data", color="orange")
    axes[0].plot(X_input, poisson_mean_output, label="Predictive Mean", color="blue")
    axes[0].fill_between(X_input, lower_bounds, upper_bounds, color="blue", alpha=0.2, label="95% CI")
    axes[0].set_title("PoLoN Process Mean Prediction")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Predicted Poisson Output")
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    # Most probable Poisson output plot
    axes[1].scatter(X, t, label="Training Data", color="orange", alpha=0.7)
    axes[1].scatter(X_input, most_probable_output, label="Most Probable Output", color="blue")
    axes[1].set_title("PoLoN Process Most Probable Output")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Most Probable Output")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    plt.subplots_adjust(wspace=0.15, bottom=0.15)
    plt.show()

    return {
        "X_input": X_input,
        "mu_values": mu_values,
        "std_values": std_values,
        "poisson_mean_output": poisson_mean_output,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "most_probable_output": most_probable_output
    }

def predict_signal_background_with_plot(X_bg, t_bg, X_signal, t_signal,
                              bounds_theta=None, theta_init=None,
                              bounds_gauss=None, trans_params0_scaled=None,
                              X_input=None):
    """
    General PoLoN + Gaussian signal prediction respecting background vs signal distinction,
    with visualization of total prediction and signal-background decomposition.

    Parameters
    ----------
    X_bg : np.ndarray
        Background input points
    t_bg : np.ndarray
        Background counts
    X_signal : np.ndarray
        Signal input points
    t_signal : np.ndarray
        Signal counts
    bounds_theta : list of tuples, optional
        Bounds for PoLoN theta. Default [(0.0001,2), (0.01,50)]
    theta_init : np.ndarray, optional
        Initial guess for theta. Default [0.05,20]
    bounds_gauss : list of tuples, optional
        Bounds for Gaussian parameters (scaled): [(A_min,A_max),(mu_min,mu_max),(sigma_min,sigma_max)]
    trans_params0_scaled : list or np.ndarray, optional
        Initial guess for Gaussian parameters (scaled)
    X_input : np.ndarray, optional
        Points at which to predict. Default: 100 points spanning all data

    Returns
    -------
    dict
        Dictionary containing optimized parameters and predicted Poisson mean
    """

    X_bg = np.array(X_bg, dtype=np.float64)
    t_bg = np.array(t_bg, dtype=np.float64)
    X_signal = np.array(X_signal, dtype=np.float64)
    t_signal = np.array(t_signal, dtype=np.float64)

    # ---- Step 1: Optimize PoLoN hyperparameters using background only ----
    if bounds_theta is None:
        bounds_theta = [(0.0001, 2), (0.01, 50.0)]
    if theta_init is None:
        theta_init = np.array([0.05, 20.0], dtype=np.float64)

    result_theta = minimize(
        neg_log_likelihood_function,
        x0=theta_init,
        args=(X_bg, t_bg),
        method='L-BFGS-B',
        bounds=bounds_theta
    )
    theta_opt = result_theta.x

    # ---- Step 2: Optimize Gaussian signal parameters using signal points ----
    if trans_params0_scaled is None:
        trans_params0_scaled = [
            1.0,                    # A
            np.mean(X_signal)/0.1,  # mu (scaled)
            0.01                    # sigma (scaled)
        ]
    if bounds_gauss is None:
        bounds_gauss = [
            (0.01, 10),                             # A
            (min(X_signal)/0.1, max(X_signal)/0.1), # mu
            (1.0, 50.0)                             # sigma
        ]

    res_gauss = minimize(
        neg_total_loglik_scaled,
        trans_params0_scaled,
        args=(X_bg, t_bg, X_signal, t_signal, theta_opt),
        method="L-BFGS-B",
        bounds=bounds_gauss
    )

    A_opt = res_gauss.x[0] * 100
    mu_opt = res_gauss.x[1] * 0.1
    sigma_opt = res_gauss.x[2] * 0.001

    # ---- Step 3: Prepare prediction points ----
    if X_input is None:
        X_input = np.linspace(min(np.min(X_bg), np.min(X_signal)),
                              max(np.max(X_bg), np.max(X_signal)), 100)

    # ---- Step 4: Compute PoLoN latent mean and variance using background ----
    Cn = compute_cov_matrix(X_bg, *theta_opt)
    lambda_n_star, _ = newton_optimization(t_bg, Cn)

    mu_values, std_values, gaussian_signal = [], [], []

    for X_new in X_input:
        K = compute_K(X_bg, X_new, *theta_opt)
        mean_lambda = K.T @ np.linalg.inv(Cn) @ lambda_n_star
        H = compute_hessian(Cn, lambda_n_star)
        var_lambda = abs(rbf_kernel_extended(X_new, X_new, *theta_opt) + 1e-5
                         - K.T @ np.linalg.inv(Cn) @ K
                         + K.T @ np.linalg.inv(Cn) @ np.linalg.inv(H) @ np.linalg.inv(Cn) @ K)

        mu_values.append(mean_lambda.item())
        std_values.append(np.sqrt(var_lambda))
        gaussian_signal.append(A_opt * np.exp(-((X_new - mu_opt)**2)/(2*sigma_opt**2)))

    mu_values = np.array(mu_values)
    std_values = np.array(std_values)
    gaussian_signal = np.array(gaussian_signal)

    # ---- Step 5: Combine PoLoN prediction and Gaussian signal ----
    poisson_mean_bg = np.exp(mu_values + std_values**2 / 2)
    poisson_mean_total = poisson_mean_bg + gaussian_signal

    # ---- Step 6: Plot results ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # (a) Total prediction vs observed
    axes[0].scatter(X_bg, t_bg, color="gray", label="Background Data", alpha=0.7)
    axes[0].scatter(X_signal, t_signal, color="orange", label="Signal Data", alpha=0.7)
    axes[0].plot(X_input, poisson_mean_total, color="blue", label="Total Prediction (Signal + Background)")
    axes[0].plot(X_input, poisson_mean_bg, color="green", linestyle="--", label="Background (PoLoN)")
    axes[0].fill_between(X_input,
                         poisson_mean_total - np.std(poisson_mean_total)*0.2,
                         poisson_mean_total + np.std(poisson_mean_total)*0.2,
                         color="blue", alpha=0.15, label="Approx. Uncertainty")
    axes[0].set_title("Total PoLoN + Gaussian Signal Prediction")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Predicted Counts")
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    # (b) Signal-background decomposition
    axes[1].plot(X_input, gaussian_signal, color="red", label="Gaussian Signal")
    axes[1].plot(X_input, poisson_mean_bg, color="green", linestyle="--", label="PoLoN Background")
    axes[1].plot(X_input, poisson_mean_total, color="blue", label="Combined Prediction")
    axes[1].set_title("Signal and Background Decomposition")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Counts")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    plt.subplots_adjust(wspace=0.15, bottom=0.15)
    plt.show()

    # ---- Step 7: Return results ----
    return {
        "theta_opt": theta_opt,
        "A_opt": A_opt,
        "mu_opt": mu_opt,
        "sigma_opt": sigma_opt,
        "X_input": X_input,
        "mu_values": mu_values,
        "std_values": std_values,
        "gaussian_signal": gaussian_signal,
        "poisson_mean_background": poisson_mean_bg,
        "poisson_mean_total": poisson_mean_total,
        "theta_success": result_theta.success,
        "gauss_success": res_gauss.success
    }
