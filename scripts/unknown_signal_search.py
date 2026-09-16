import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, poisson

from helper import *


# ============================================================
# 1. SIGMOID FUNCTION
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# 2. PENALIZED NEGATIVE LOG-LIKELIHOOD
# ============================================================

def neg_log_likelihood_penalized(theta, X, t, l):
    """
    Penalized negative log-likelihood used for the
    background Gaussian-process hyperparameter optimization.
    """

    Cn = compute_cov_matrix(X, *theta)

    lambda_n_star, _ = newton_optimization(t, Cn)

    H = compute_hessian(Cn, lambda_n_star)

    log_likelihood_value = (
        t.T @ lambda_n_star
        - np.sum(np.exp(lambda_n_star))
        - 0.5 * log_det_cholesky(H)
        + multivariate_normal.logpdf(
            lambda_n_star,
            mean=np.zeros(X.shape[0]),
            cov=Cn
        )
    ) + l * sigmoid(10 * (theta[0] - 0.1))

    return -log_likelihood_value


# ============================================================
# 3. SYNTHETIC DATA GENERATION
# ============================================================

def generate_data(
    signal_strength,
    n_instances_per_X,
    seed,
    signal_location,
    std_gaussian=0.004
):
    """
    Generate synthetic Poisson count data containing
    an exponentially decreasing background and a Gaussian signal.

    Parameters
    ----------
    signal_strength : float
        True amplitude of the Gaussian signal.

    n_instances_per_X : int
        Number of Poisson observations generated at each X value.

    seed : int
        Random seed.

    signal_location : float
        True location of the Gaussian signal.

    std_gaussian : float, optional
        True width of the Gaussian signal.
        Default is 0.004.

    Returns
    -------
    X : ndarray
        Expanded X values.

    t : ndarray
        Generated Poisson count observations.
    """

    np.random.seed(seed)

    c = np.exp(11.7)
    lambda_exp = 30.6

    n_samples = 40

    X_unique = np.linspace(0.1, 0.16, n_samples)

    X_expanded = []
    t_expanded = []

    for x in X_unique:

        # Background rate
        alpha_x = c * np.exp(-lambda_exp * x)

        # Gaussian signal
        gaussian_signal = (
            signal_strength
            * np.exp(
                -((x - signal_location) ** 2)
                / (2 * std_gaussian ** 2)
            )
        )

        # Total expected rate
        alpha_noisy = alpha_x + gaussian_signal

        # Poisson observations
        t_values = np.random.poisson(
            alpha_noisy,
            n_instances_per_X
        )

        X_expanded.extend([x] * n_instances_per_X)
        t_expanded.extend(t_values)

    return np.array(X_expanded), np.array(t_expanded)


# ============================================================
# 4. UNKNOWN-LOCATION SIGNAL SEARCH
# ============================================================

def unknown_signal_search(
    X,
    t,
    std_gaussian=0.004,
    window_size=20,
    penalty=2000
):
    """
    Search for a Gaussian signal whose location and amplitude
    are unknown.

    Parameters
    ----------
    X : ndarray
        Observed X values.

    t : ndarray
        Observed Poisson counts.

    std_gaussian : float, optional
        Gaussian signal width used in the signal model.
        Default is 0.004.

    window_size : int, optional
        Number of unique X values included in each sliding window.
        Default is 20.

    penalty : float, optional
        Penalty parameter used in the background hyperparameter
        optimization. Default is 2000.

    Returns
    -------
    results : pandas.DataFrame
        Results from every sliding-window position.
    """

    # Get unique X values directly from the supplied data
    X_unique = np.unique(X)

    results = []

    # --------------------------------------------------------
    # Loop over all possible sliding windows
    # --------------------------------------------------------

    for i in range(len(X_unique) - window_size + 1):

        window_x = X_unique[i:i + window_size]

        center = np.mean(window_x)

        # ----------------------------------------------------
        # Separate signal window and background region
        # ----------------------------------------------------

        mask_sig = np.isin(X, window_x)
        mask_bg = ~mask_sig

        X_bg = X[mask_bg]
        t_bg = t[mask_bg]

        X_signal = X[mask_sig]
        t_signal = t[mask_sig]

        # ----------------------------------------------------
        # Background-only hyperparameter optimization
        # ----------------------------------------------------

        bounds = [
            (0.0001, 3),
            (0.01, 50.0)
        ]

        theta_init = np.array(
            [0.05, 20],
            dtype=np.float64
        )

        result = minimize(
            neg_log_likelihood_penalized,
            x0=theta_init,
            args=(X_bg, t_bg, penalty),
            method="L-BFGS-B",
            bounds=bounds
        )

        theta_hat = result.x

        # ----------------------------------------------------
        # Signal + background optimization
        # ----------------------------------------------------
        #
        # The signal amplitude and location are unknown.
        #
        # initial_amplitude is only a numerical starting value
        # for the optimizer; it is NOT treated as the known
        # signal strength.
        # ----------------------------------------------------

        initial_amplitude = 250

        trans_params0_scaled = [
            initial_amplitude / 100,
            center / 0.1,
            std_gaussian / 0.001
        ]

        mu_min_scaled = window_x[0] / 0.1
        mu_max_scaled = window_x[-1] / 0.1

        bounds_scaled = [
            (0, 6),
            (mu_min_scaled, mu_max_scaled),
            (1.0, 7.0)
        ]

        res = minimize(
            neg_total_loglik_scaled,
            trans_params0_scaled,
            args=(
                X_bg,
                t_bg,
                X_signal,
                t_signal,
                theta_hat
            ),
            method="L-BFGS-B",
            bounds=bounds_scaled
        )

        # Convert scaled parameters back to physical values
        A_hat, mu_hat, sigma_hat = res.x

        A_hat *= 100
        mu_hat *= 0.1
        sigma_hat *= 0.001

        # ----------------------------------------------------
        # Background GP prediction
        # ----------------------------------------------------

        Cn_bg = compute_cov_matrix(
            X_bg,
            *theta_hat
        )

        lambda_n_star, _ = newton_optimization(
            t_bg,
            Cn_bg
        )

        # ----------------------------------------------------
        # Background-only likelihood
        # ----------------------------------------------------

        loglik_bg, _ = signal_loglik_and_hessian(
            X_bg,
            t_signal,
            X_signal,
            lambda_n_star,
            theta_hat,
            0,
            mu_hat,
            sigma_hat
        )

        # ----------------------------------------------------
        # Signal + background likelihood
        # ----------------------------------------------------

        loglik_signal, _ = signal_loglik_and_hessian(
            X_bg,
            t_signal,
            X_signal,
            lambda_n_star,
            theta_hat,
            A_hat,
            mu_hat,
            sigma_hat
        )

        # ----------------------------------------------------
        # Total likelihood
        # ----------------------------------------------------

        _, _, loglik_tot = total_loglik_and_hessian(
            X_bg,
            t_bg,
            X_signal,
            t_signal,
            theta_hat,
            A_hat,
            mu_hat,
            sigma_hat
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        row = {
            "window_index": i,
            "window_center": center,
            "x_min": window_x[0],
            "x_max": window_x[-1],
            "ll_bg": loglik_bg,
            "ll_signal": loglik_signal,
            "A_hat": A_hat,
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "theta0": theta_hat[0],
            "theta1": theta_hat[1],
            "N_signal": len(t_signal),
            "N_bg": len(t_bg),
            "ll_tot": loglik_tot
        }

        results.append(row)

        print(
            f"Window {i} done, "
            f"center={center:.5f}"
        )

    # --------------------------------------------------------
    # Return all windows as a DataFrame
    # --------------------------------------------------------

    return pd.DataFrame(results)
