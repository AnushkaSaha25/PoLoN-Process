import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import argparse
import os
from helpers import *


# ============================================================
# 1. SIGMOID + PENALIZED LIKELIHOOD
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def neg_log_likelihood_penalized(theta, X, T, Nv, l):
    C           = compute_cov_matrix(X, *theta)
    lam_star, H = newton_optimization(T, Nv, C)
    log_lik = (
        T @ lam_star
        - np.sum(Nv * np.exp(lam_star))
        - 0.5 * log_det_cholesky(H)
        + multivariate_normal.logpdf(
            lam_star,
            mean=np.zeros(X.shape[0]),
            cov=C
        )
    ) + l * sigmoid(10 * (theta[0] - 0.1))
    return -log_lik


# ============================================================
# 2. DATA GENERATION
# ============================================================

def generate_data(signal_strength, n_instances_per_X, seed, signal_location):
    np.random.seed(seed)
    c          = np.exp(11.7)
    lambda_exp = 30.6
    n_samples  = 40
    X_unique   = np.linspace(0.1, 0.16, n_samples)
    std_gaussian = 0.004

    X_expanded, t_expanded = [], []
    for x in X_unique:
        alpha_x         = c * np.exp(-lambda_exp * x)
        gaussian_signal = signal_strength * np.exp(
            -((x - signal_location)**2) / (2 * std_gaussian**2)
        )
        alpha_total = alpha_x + gaussian_signal
        t_vals      = np.random.poisson(alpha_total, n_instances_per_X)
        X_expanded.extend([x] * n_instances_per_X)
        t_expanded.extend(t_vals)
    X_out = np.round(np.array(X_expanded, dtype=float), 10).reshape(-1, 1)
    t_out = np.array(t_expanded, dtype=float)
    return X_out, t_out, X_unique, signal_location, std_gaussian


# ============================================================
# 3. SCAN
# ============================================================

def run_scan(
    signal_strength    = 250,
    n_instances_per_X  = 5,
    seed               = 42,
    window_size        = 20,
    penalty            = 2000,
    signal_location    = 0.135,
    output_file        = "scan_results.csv"
):
    X_raw, t_raw, X_unique, mean_gaussian, std_gaussian = generate_data(
        signal_strength, n_instances_per_X, seed, signal_location
    )

    # ---- initialise CSV ----
    if os.path.exists(output_file):
        os.remove(output_file)

    columns = ["window_index", "window_center", "x_min", "x_max",
               "ll_bg", "ll_signal", "TS",
               "A_hat", "mu_hat", "sigma_hat",
               "theta0", "theta1", "N_signal", "N_bg", "ll_tot"]
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    for i in range(len(X_unique) - window_size + 1):

        window_x = X_unique[i: i + window_size]
        center   = np.mean(window_x)

        # ---- split raw arrays ----
        mask_sig = np.array([any(np.isclose(x, window_x, atol=1e-10)) for x in X_raw[:, 0]])
        mask_bg   = ~mask_sig

        X_bg_raw  = X_raw[mask_bg]
        t_bg_raw  = t_raw[mask_bg]
        X_sig_raw = X_raw[mask_sig]
        t_sig_raw = t_raw[mask_sig]

        # ---- aggregate for Step 1 ----
        X_bg, T_bg, Nv_bg = aggregate_replicates(X_bg_raw, t_bg_raw)

        # ---- Step 1: background hyperparameter optimization ----
        theta_init   = np.array([0.05, 20.0], dtype=np.float64)
        bounds_theta = [(0.0001, 3), (0.01, 50.0)]

        result = minimize(
            neg_log_likelihood_penalized,
            x0=theta_init,
            args=(X_bg, T_bg, Nv_bg, penalty),
            method='L-BFGS-B',
            bounds=bounds_theta
        )
        theta_hat = result.x

        # ---- Step 2: signal parameter optimization ----
        trans_params0_scaled = [
            250   / 100,
            center / 0.1,
            std_gaussian / 0.001
        ]

        mu_min_scaled = window_x[0]  / 0.1
        mu_max_scaled = window_x[-1] / 0.1

        bounds_scaled = [
            (0, 6),
            (mu_min_scaled, mu_max_scaled),
            (1.0, 7.0)
        ]

        res = minimize(
            neg_total_loglik_scaled,
            trans_params0_scaled,
            args=(X_bg_raw, t_bg_raw, X_sig_raw, t_sig_raw, theta_hat),
            method="L-BFGS-B",
            bounds=bounds_scaled
        )

        A_hat     = res.x[0] * 100
        mu_hat    = res.x[1] * 0.1
        sigma_hat = res.x[2] * 0.001

        # ---- compute log-likelihoods ----
        C_bg        = compute_cov_matrix(X_bg, sigma=theta_hat[0],
                                          gamma=theta_hat[1], regularization=1e-6)
        lam_star, _ = newton_optimization(T_bg, Nv_bg, C_bg)

        # background-only at signal locations (A=0)
        loglik_bg = signal_loglik(
            X_bg_raw, t_bg_raw, t_sig_raw, X_sig_raw,
            lam_star, theta_hat, 0.0, mu_hat, sigma_hat
        )

        # signal + background at signal locations
        loglik_signal = signal_loglik(
            X_bg_raw, t_bg_raw, t_sig_raw, X_sig_raw,
            lam_star, theta_hat, A_hat, mu_hat, sigma_hat
        )

        # total log-likelihood
        _, _, loglik_tot = total_loglik(
            X_bg_raw, t_bg_raw, X_sig_raw, t_sig_raw,
            theta_hat, A_hat, mu_hat, sigma_hat
        )

        # ---- test statistic ----
        TS = 2 * (loglik_signal - loglik_bg)

        # ---- save row ----
        row = {
            "window_index"  : i,
            "window_center" : center,
            "x_min"         : window_x[0],
            "x_max"         : window_x[-1],
            "ll_bg"         : loglik_bg,
            "ll_signal"     : loglik_signal,
            "TS"            : TS,
            "A_hat"         : A_hat,
            "mu_hat"        : mu_hat,
            "sigma_hat"     : sigma_hat,
            "theta0"        : theta_hat[0],
            "theta1"        : theta_hat[1],
            "N_signal"      : len(t_sig_raw),
            "N_bg"          : len(t_bg_raw),
            "ll_tot"        : loglik_tot
        }
        pd.DataFrame([row]).to_csv(output_file, mode='a', header=False, index=False)
        print(f"Window {i} done, center={center:.5f}, TS={TS:.2f}")

    print(f"Scan completed. Results saved to {output_file}")


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoLoN-SB sliding window scan")
    parser.add_argument("--signal_strength",   type=float, default=50)
    parser.add_argument("--n_instances_per_X", type=int,   default=20)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--window_size",       type=int,   default=20)
    parser.add_argument("--signal_location",   type=float, default=0.135)
    parser.add_argument("--output_file",       type=str,   default="scan_results.csv")
    args = parser.parse_args()

    run_scan(
        signal_strength   = args.signal_strength,
        n_instances_per_X = args.n_instances_per_X,
        seed              = args.seed,
        window_size       = args.window_size,
        signal_location   = args.signal_location,
        output_file       = args.output_file
    )
