import numpy as np
from scipy.stats import multivariate_normal,poisson
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Kernel and covariance
# ---------------------------------------------------------------------------

def rbf_kernel(x, y, sigma=1.0, gamma=1.0):
    dist_squared = np.sum((x - y) ** 2)
    return gamma * np.exp(-dist_squared / (2 * sigma ** 2))


def compute_cov_matrix(X, sigma=1.0, gamma=1.0, regularization=1e-5):
    """
    M x M covariance matrix over the M unique input locations.
    X : (M, d) array of unique input locations.
    """
    M = X.shape[0]
    C = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            C[i, j] = rbf_kernel(X[i], X[j], sigma, gamma)
    C += regularization * np.eye(M)
    return C


def compute_k_vec(X, X_new, sigma=1.0, gamma=1.0):
    """
    Length-M vector of kernel values between each training location and X_new.
    """
    M = X.shape[0]
    k = np.zeros(M)
    for i in range(M):
        k[i] = rbf_kernel(X[i], X_new, sigma, gamma)
    return k


# ---------------------------------------------------------------------------
# Sufficient statistics: aggregate replicates at each location
# ---------------------------------------------------------------------------

def aggregate_replicates(X_all, t_all):
    """
    Given N total observations (with repetitions), compute per-location
    sufficient statistics.

    Parameters
    ----------
    X_all : (N, d) array — input location for each observation (rows may repeat)
    t_all : (N,)   array — integer count for each observation

    Returns
    -------
    X  : (M, d) array — unique input locations
    T  : (M,)   array — total count at each location, T_n = sum_i t_{n,i}
    Nv : (M,)   array — number of replicates at each location, N_n
    """
    # find unique locations and the inverse index
    X, inv = np.unique(X_all, axis=0, return_inverse=True)
    M = X.shape[0]
    T  = np.zeros(M)
    Nv = np.zeros(M)
    for n in range(M):
        mask   = (inv == n)
        T[n]  = t_all[mask].sum()
        Nv[n] = mask.sum()
    return X, T, Nv


# ---------------------------------------------------------------------------
# Gradient and Hessian of Psi(lambda_M)
# ---------------------------------------------------------------------------

def compute_residual(T, Nv, lambda_vector):
    """
    D_n = T_n - N_n * exp(lambda_n)   (M-vector)
    """
    return T - Nv * np.exp(lambda_vector)


def compute_gradient(T, Nv, lambda_vector, C):
    """
    grad Psi = D - C^{-1} lambda
    """
    D     = compute_residual(T, Nv, lambda_vector)
    C_inv = np.linalg.inv(C)
    return D - C_inv @ lambda_vector


def compute_hessian(C, lambda_vector, Nv):
    """
    H = C^{-1} + W_M,   W_M = diag(N_n * exp(lambda_n))
    """
    C_inv = np.linalg.inv(C)
    W     = np.diag(Nv * np.exp(lambda_vector))
    return C_inv + W


# ---------------------------------------------------------------------------
# Newton-Raphson to find lambda_M^star
# ---------------------------------------------------------------------------

def newton_optimization(T, Nv, C, max_iter=20000, tol=1e-6):
    """
    Solve C^{-1} lambda^* = D(lambda^*)  via Newton-Raphson.
    Each Newton step is damped by a factor 0.001. Iteration stops when max|change| < tol (default 1e-6) or after max_iter (default 20000) steps.
    Parameters
    ----------
    T  : (M,) total counts per location
    Nv : (M,) replicate counts per location
    C  : (M, M) covariance matrix
    """
    lam = np.ones(C.shape[0])
    for _ in range(max_iter):
        grad   = compute_gradient(T, Nv, lam, C)
        H      = compute_hessian(C, lam, Nv)
        change = 0.001 * np.linalg.solve(H, grad)
        lam   += change
        if np.max(np.abs(change)) < tol:
            break
    return lam, compute_hessian(C, lam, Nv)


# ---------------------------------------------------------------------------
# Log-determinant via Cholesky
# ---------------------------------------------------------------------------

def log_det_cholesky(matrix):
    L = np.linalg.cholesky(matrix)
    return 2.0 * np.sum(np.log(np.diag(L)))


# ---------------------------------------------------------------------------
# Laplace-approximated marginal log-likelihood (for hyperparameter optimisation)
# ---------------------------------------------------------------------------

def neg_log_likelihood(theta, X, T, Nv):
    """
    Negative Laplace-approximated marginal log-likelihood.

    log p(t_N | theta) ≈ Psi(lambda^*) - 0.5 * log|I_M + C W| + const

    which is equivalent (up to constants independent of theta) to:
        T^T lambda^*  -  sum_n N_n exp(lambda_n^*)
        - 0.5 * log|H|
        - 0.5 * log|C|
        + log p(lambda^* | theta)          <- = -0.5 lambda^{*T} C^{-1} lambda^*  + const

    Parameters
    ----------
    theta : kernel hyperparameters (sigma, gamma)
    X     : (M, d) unique locations
    T     : (M,)   total counts
    Nv    : (M,)   replicate counts
    """
    C               = compute_cov_matrix(X, *theta)
    lam_star, H     = newton_optimization(T, Nv, C)

    log_lik = (
        T @ lam_star
        - np.sum(Nv * np.exp(lam_star))
        - 0.5 * log_det_cholesky(H)
        + multivariate_normal.logpdf(
            lam_star,
            mean=np.zeros(X.shape[0]),
            cov=C
          )
    )
    return -log_lik


# ---------------------------------------------------------------------------
# Hyperparameter optimisation via L-BFGS-B
# ---------------------------------------------------------------------------

def optimise_hyperparameters(X, T, Nv, theta_init, bounds=None):
    """
    Maximise the Laplace marginal log-likelihood over kernel hyperparameters.

    Parameters
    ----------
    X          : (M, d) unique locations
    T          : (M,)   total counts
    Nv         : (M,)   replicate counts
    theta_init : initial guess for (sigma, gamma)
    bounds     : list of (min, max) pairs for each hyperparameter
                 default: [(1e-3, None), (1e-3, None)]  (both positive)
    """
    if bounds is None:
        bounds = [(1e-3, None), (1e-3, None)]

    result = minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        args=(X, T, Nv),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    return result.x


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(X_new, X, T, Nv, theta, c_reg=1e-5, lam_star=None):
    """
    Predictive distribution parameters for a new point X_new.

    Handles both cases:
      Case A: X_new is a new location not in X  ->  GP conditional
      Case B: X_new coincides with training location X[m]  ->  Laplace marginal

    Returns
    -------
    mu    : predictive mean of log-rate
    sigma2: predictive variance of log-rate
    case  : 'A' or 'B'
    """
    sigma, gamma = theta
    C        = compute_cov_matrix(X, sigma, gamma, regularization=c_reg)
    if lam_star is None:
        lam_star, H = newton_optimization(T, Nv, C)
    else:                                   # reuse an already-solved lam_star
        H = compute_hessian(C, lam_star, Nv)

    # check whether X_new matches a training location
    dists = np.linalg.norm(X - X_new, axis=1)
    match = np.where(dists < 1e-10)[0]

    if len(match) > 0:
        # Case B: training location
        m      = match[0]
        H_inv  = np.linalg.inv(H)
        mu     = lam_star[m]
        sigma2 = H_inv[m, m]
        return mu, sigma2, 'B'
    else:
        # Case A: new location
        k     = compute_k_vec(X, X_new, sigma, gamma)
        c     = rbf_kernel(X_new, X_new, sigma, gamma) + c_reg
        C_inv = np.linalg.inv(C)
        mu    = k @ C_inv @ lam_star
        # sigma^2 = c - k^T (C + W^{-1})^{-1} k
        W_inv  = np.diag(1.0 / (Nv * np.exp(lam_star)))
        sigma2 = c - k @ np.linalg.inv(C + W_inv) @ k
        return mu, sigma2, 'A'


# ---------------------------------------------------------------------------
# Signal function
# ---------------------------------------------------------------------------

def gaussian_bump(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))


def background_loglik(X_bg_raw, t_bg_raw, lam_star, A, mu, sigma):
    X_unique, inv = np.unique(X_bg_raw, axis=0, return_inverse=True)
    G    = gaussian_bump(X_unique[:, 0], A, mu, sigma)  # force 1D
    rate = np.exp(lam_star) + G                          # (M_bg,)
    rate_expanded = rate[inv]                            # (M_bg * Np,)
    return np.sum(poisson.logpmf(t_bg_raw.astype(int), rate_expanded))


def signal_loglik(X_bg_raw, t_bg_raw, t_signal_raw, X_signal_raw,
                  lam_star, theta, A, mu, sigma):
    sigma_k, gamma_k = theta
    X_bg_unique,  _       = np.unique(X_bg_raw,     axis=0, return_inverse=True)
    X_sig_unique, inv_sig = np.unique(X_signal_raw, axis=0, return_inverse=True)
    _, Nv_bg = np.unique(X_bg_raw, axis=0, return_counts=True)

    C_bg  = compute_cov_matrix(X_bg_unique, sigma=sigma_k, gamma=gamma_k,
                                regularization=1e-6)
    C_inv = np.linalg.inv(C_bg)
    H_gp  = compute_hessian(C_bg, lam_star, Nv_bg.astype(float))
    H_inv = np.linalg.inv(H_gp)

    rates_unique = []
    for idx in range(len(X_sig_unique)):
        x_new    = X_sig_unique[idx].reshape(1, -1)
        k        = compute_k_vec(X_bg_unique, x_new, sigma=sigma_k, gamma=gamma_k)
        c_val    = float(rbf_kernel(x_new, x_new,
                                     sigma=sigma_k, gamma=gamma_k)) + 1e-5
        L        = k @ C_inv
        mu_pred  = float(L @ lam_star)
        var_pred = float(abs(c_val - L @ k + L @ H_inv @ L.T + 1e-5))
        mean_log = np.clip(mu_pred + var_pred / 2.0, -500, 500)
        g        = float(gaussian_bump(X_sig_unique[idx, 0], A, mu, sigma))  # force scalar
        rate     = np.exp(mean_log) + g
        rates_unique.append(rate if np.isfinite(rate) and rate > 0 else 1e-10)

    rates_unique  = np.array(rates_unique)
    rate_expanded = rates_unique[inv_sig]   # (M_sig * Np,)
    return np.sum(poisson.logpmf(t_signal_raw.astype(int), rate_expanded))


def total_loglik(X_bg_raw, t_bg_raw, X_signal_raw, t_signal_raw,
                 theta, A, mu, sigma):
    X_bg, T_bg, Nv_bg = aggregate_replicates(X_bg_raw, t_bg_raw)

    sigma_k, gamma_k = theta
    C_bg        = compute_cov_matrix(X_bg, sigma=sigma_k, gamma=gamma_k,
                                      regularization=1e-6)
    lam_star, _ = newton_optimization(T_bg, Nv_bg, C_bg)

    ll_bg  = background_loglik(X_bg_raw, t_bg_raw,
                                lam_star, A, mu, sigma)
    ll_sig = signal_loglik(X_bg_raw, t_bg_raw, t_signal_raw, X_signal_raw,
                           lam_star, theta, A, mu, sigma)
    return ll_bg, ll_sig, ll_sig + ll_bg


def neg_total_loglik_scaled(params_scaled, X_bg_raw, t_bg_raw,
                             X_signal_raw, t_signal_raw, theta):
    A_scaled, mu_scaled, sigma_scaled = params_scaled
    A     = A_scaled     * 100
    mu    = mu_scaled    * 0.1
    sigma = sigma_scaled * 0.001
    _, _, tot_log_lik = total_loglik(
        X_bg_raw, t_bg_raw,
        X_signal_raw, t_signal_raw,
        theta, A, mu, sigma
    )
    return -tot_log_lik


# ===========================================================================
# PREDICTION AND PLOTTING  (added: predict_many, polon_*, predict_signal_background_*, plot_*)
# ===========================================================================

def predict_many(X, T, Nv, theta, X_new, lam_star=None, c_reg=1e-5):
    """
    Apply predict() to several inputs.

    New locations use the GP conditional (Case A) and training locations the
    Laplace marginal (Case B), exactly as predict() does. Pass lam_star (from
    newton_optimization) so the Newton solve is done once instead of per point.

    X_new : (P, d) or (P,) prediction inputs
    Returns mu, std: (P,) arrays, the mean and standard deviation of the latent
    log-rate. The PLN predictive mean is exp(mu + std**2 / 2).
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)
    if lam_star is None:
        lam_star, _ = newton_optimization(
            T, Nv, compute_cov_matrix(X, theta[0], theta[1], regularization=c_reg))
    mu = np.empty(len(X_new))
    std = np.empty(len(X_new))
    for i, x_new in enumerate(X_new):
        mu[i], sigma2, _ = predict(x_new, X, T, Nv, theta, c_reg=c_reg, lam_star=lam_star)
        std[i] = np.sqrt(abs(sigma2))
    return mu, std


# ---------------------------------------------------------------------------
# PoLoN
# ---------------------------------------------------------------------------

def polon_predict(X_train, t_train, X_input=None, theta_init=None, bounds=None,
                  n_samples=5000, theta=None):
    """
    Fit PoLoN and predict. No plotting. Returns a results dict that
    plot_polon_prediction() can draw without refitting.

    X_train, t_train : raw observations (repeated inputs allowed)
    theta            : optional (sigma, gamma). If given, the hyperparameter
                       optimisation is skipped and these values are used.
    """
    X_raw = np.asarray(X_train, dtype=np.float64)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)
    t_raw = np.asarray(t_train, dtype=np.float64)

    X, T, Nv = aggregate_replicates(X_raw, t_raw)

    theta_success = None
    if theta is None:
        if theta_init is None:
            theta_init = np.array([0.01, 20.0], dtype=np.float64)
        if bounds is None:
            bounds = [(0.0001, 30), (0.01, 30)]
        result = minimize(neg_log_likelihood, x0=theta_init, args=(X, T, Nv),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12})  # as optimise_hyperparameters
        theta_opt, theta_success = result.x, result.success
        if result.nit == 0:
            print("Warning: the optimiser stopped before taking a step, so theta is "
                  "still the initial guess. Try a different theta_init.")
        print(f"Optimized theta: {theta_opt}")
        print(f"Log-likelihood at optimum: {-result.fun:.4f}")
    else:
        theta_opt = np.asarray(theta, dtype=np.float64)
        print(f"Using supplied theta: {theta_opt}")

    C = compute_cov_matrix(X, *theta_opt)
    lam_star, _ = newton_optimization(T, Nv, C, max_iter=25000, tol=1e-5)

    if X_input is None:
        if X.shape[1] != 1:
            raise ValueError("X_input must be given when X_train has more than one feature.")
        X_input = np.linspace(X.min(), X.max(), 100)
    X_in = np.asarray(X_input, dtype=np.float64)
    if X_in.ndim == 1:
        X_in = X_in.reshape(-1, 1)

    mu_values, std_values = predict_many(X, T, Nv, theta_opt, X_in, lam_star=lam_star)
    poisson_mean_output = np.exp(mu_values + std_values ** 2 / 2)

    # Monte Carlo: most probable count and 95% interval of the Poisson log-normal mixture
    k_range = np.arange(0, int(np.max(poisson_mean_output) * 2) + 10)
    most_probable_output, lower_bounds, upper_bounds = [], [], []
    for mu, s in zip(mu_values, std_values):
        lam_samples = np.random.lognormal(mean=mu, sigma=s, size=n_samples)
        probs = np.array([np.mean(poisson.pmf(k, lam_samples)) for k in k_range])
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]                      # k_range is truncated, so normalise
        last = len(k_range) - 1
        most_probable_output.append(k_range[np.argmax(probs)])
        lower_bounds.append(k_range[min(np.searchsorted(cdf, 0.025), last)])
        upper_bounds.append(k_range[min(np.searchsorted(cdf, 0.975), last)])

    return {
        "theta_opt": theta_opt,
        "theta_success": theta_success,
        "X_train": X_raw, "t_train": t_raw,
        "X_input": X_in[:, 0] if X_in.shape[1] == 1 else X_in,
        "mu_values": mu_values,
        "std_values": std_values,
        "poisson_mean_output": poisson_mean_output,
        "lower_bounds": np.array(lower_bounds),
        "upper_bounds": np.array(upper_bounds),
        "most_probable_output": np.array(most_probable_output),
    }


def plot_polon_prediction(results, savepath=None, show=True):
    """Plot the output of polon_predict (1D inputs only). Does not refit."""
    import matplotlib.pyplot as plt
    X_raw, t_raw = results["X_train"], results["t_train"]
    if X_raw.shape[1] != 1:
        raise ValueError("plot_polon_prediction supports 1D inputs only.")
    x_plot = np.asarray(results["X_input"]).reshape(len(results["X_input"]), -1)[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(X_raw[:, 0], t_raw, label="Training Data", color="orange")
    axes[0].plot(x_plot, results["poisson_mean_output"], label="Predictive Mean", color="blue")
    axes[0].fill_between(x_plot, results["lower_bounds"], results["upper_bounds"],
                         color="blue", alpha=0.2, label="95% CI")
    axes[0].set_title("PoLoN Process Mean Prediction")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Predicted Poisson Output")
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    axes[1].scatter(X_raw[:, 0], t_raw, label="Training Data", color="orange", alpha=0.7)
    axes[1].scatter(x_plot, results["most_probable_output"], label="Most Probable Output", color="blue")
    axes[1].set_title("PoLoN Process Most Probable Output")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Most Probable Output")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    plt.subplots_adjust(wspace=0.15, bottom=0.15)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def polon_predict_and_plot(X_train, t_train, X_input=None, theta_init=None,
                           bounds=None, n_samples=5000, plot=True, theta=None):
    """Predict, then plot (1D inputs). Returns the results dict."""
    results = polon_predict(X_train, t_train, X_input=X_input, theta_init=theta_init,
                            bounds=bounds, n_samples=n_samples, theta=theta)
    if plot and results["X_train"].shape[1] == 1:
        plot_polon_prediction(results)
    return results


# ---------------------------------------------------------------------------
# PoLoN-SB (specified signal window)
# ---------------------------------------------------------------------------

def predict_signal_background(X_bg, t_bg, X_signal, t_signal,
                              bounds_theta=None, theta_init=None,
                              bounds_signal=None, signal_params0_scaled=None,
                              X_input=None, n_samples=5000,
                              theta=None, signal_params=None):
    """
    Fit PoLoN background + Gaussian signal and predict. No plotting.
    Inputs are raw observations (repeated inputs allowed).

    theta         : optional (sigma, gamma); skips the background optimisation.
    signal_params : optional (A, mu, sigma) in physical units; skips the signal
                    optimisation.
    """
    X_bg_raw = np.asarray(X_bg, dtype=np.float64)
    X_sig_raw = np.asarray(X_signal, dtype=np.float64)
    if X_bg_raw.ndim == 1:
        X_bg_raw = X_bg_raw.reshape(-1, 1)
    if X_sig_raw.ndim == 1:
        X_sig_raw = X_sig_raw.reshape(-1, 1)
    t_bg_raw = np.asarray(t_bg, dtype=np.float64)
    t_sig_raw = np.asarray(t_signal, dtype=np.float64)

    X_u, T_bg, Nv_bg = aggregate_replicates(X_bg_raw, t_bg_raw)

    # ---- Step 1: background hyperparameters ----
    theta_success = None
    if theta is None:
        if bounds_theta is None:
            bounds_theta = [(0.0001, 2), (0.01, 50.0)]
        if theta_init is None:
            theta_init = np.array([0.05, 20.0], dtype=np.float64)
        result_theta = minimize(neg_log_likelihood, x0=theta_init, args=(X_u, T_bg, Nv_bg),
                                method='L-BFGS-B', bounds=bounds_theta,
                                options={'maxiter': 1000, 'ftol': 1e-12})  # as optimise_hyperparameters
        theta_opt, theta_success = result_theta.x, result_theta.success
    else:
        theta_opt = np.asarray(theta, dtype=np.float64)

    # ---- Step 2: signal parameters ----
    signal_success = None
    if signal_params is None:
        x_sig_1d = X_sig_raw[:, 0]
        if signal_params0_scaled is None:
            signal_params0_scaled = [1.0, np.mean(x_sig_1d) / 0.1, 4.0]
        if bounds_signal is None:
            bounds_signal = [(0.01, 10),
                            (x_sig_1d.min() / 0.1, x_sig_1d.max() / 0.1),
                            (1.0, 50.0)]
        res_gauss = minimize(neg_total_loglik_scaled, signal_params0_scaled,
                             args=(X_bg_raw, t_bg_raw, X_sig_raw, t_sig_raw, theta_opt),
                             method="L-BFGS-B", bounds=bounds_signal)
        A_opt = res_gauss.x[0] * 100
        mu_opt = res_gauss.x[1] * 0.1
        sigma_opt = res_gauss.x[2] * 0.001
        signal_success = res_gauss.success
    else:
        A_opt, mu_opt, sigma_opt = (float(v) for v in signal_params)

    # ---- Step 3: prediction ----
    if X_input is None:
        lo = min(X_bg_raw.min(), X_sig_raw.min())
        hi = max(X_bg_raw.max(), X_sig_raw.max())
        X_input = np.linspace(lo, hi, 100)
    X_in = np.asarray(X_input, dtype=np.float64)
    if X_in.ndim == 1:
        X_in = X_in.reshape(-1, 1)

    C = compute_cov_matrix(X_u, *theta_opt)
    lam_star, _ = newton_optimization(T_bg, Nv_bg, C)
    mu_values, std_values = predict_many(X_u, T_bg, Nv_bg, theta_opt, X_in, lam_star=lam_star)

    signal_component = gaussian_bump(X_in[:, 0], A_opt, mu_opt, sigma_opt)
    poisson_mean_bg = np.exp(mu_values + std_values ** 2 / 2)
    poisson_mean_total = poisson_mean_bg + signal_component

    # 95% interval: lognormal background rate + signal, then Poisson counts
    lam_s = np.random.lognormal(mean=mu_values, sigma=std_values,
                                size=(n_samples, len(mu_values))) + signal_component
    counts = np.random.poisson(lam_s)
    lower_bounds, upper_bounds = np.percentile(counts, [2.5, 97.5], axis=0)

    # most probable count: mode of the mixture (1/n) * sum_s Poisson(k; lam_s), searched
    # over the range the sampled counts actually cover (0.1% to 99.9%)
    most_probable_output = np.empty(len(mu_values))
    for i in range(len(mu_values)):
        k_lo, k_hi = np.percentile(counts[:, i], [0.1, 99.9])
        k = np.arange(int(np.floor(k_lo)), int(np.ceil(k_hi)) + 1)
        pmf = np.concatenate([                      # chunked over k to bound memory
            poisson.pmf(k[j:j + 256, None], lam_s[:, i][None, :]).mean(axis=1)
            for j in range(0, len(k), 256)])
        most_probable_output[i] = k[np.argmax(pmf)]

    return {
        "theta_opt": theta_opt,
        "A_opt": A_opt, "mu_opt": mu_opt, "sigma_opt": sigma_opt,
        "X_bg": X_bg_raw, "t_bg": t_bg_raw,
        "X_signal": X_sig_raw, "t_signal": t_sig_raw,
        "X_input": X_in[:, 0] if X_in.shape[1] == 1 else X_in,
        "mu_values": mu_values, "std_values": std_values,
        "signal_component": signal_component,
        "poisson_mean_background": poisson_mean_bg,
        "poisson_mean_total": poisson_mean_total,
        "lower_bounds": lower_bounds, "upper_bounds": upper_bounds,
        "most_probable_output": most_probable_output,
        "theta_success": theta_success,
        "signal_success": signal_success,
    }


def plot_signal_background(results, savepath=None, show=True):
    """Plot the output of predict_signal_background (1D inputs only). Does not refit."""
    import matplotlib.pyplot as plt
    if np.ndim(results["X_input"]) > 1 and results["X_input"].shape[1] != 1:
        raise ValueError("plot_signal_background supports 1D inputs only.")
    x_plot = np.asarray(results["X_input"]).reshape(len(results["X_input"]), -1)[:, 0]

    X_data = np.concatenate([results["X_bg"][:, 0], results["X_signal"][:, 0]])
    t_data = np.concatenate([results["t_bg"], results["t_signal"]])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(X_data, t_data, label="Data", color="orange")
    axes[0].plot(x_plot, results["poisson_mean_total"], color="blue",
                 label="Total Prediction (Signal + Background)")
    axes[0].plot(x_plot, results["poisson_mean_background"], color="green",
                 linestyle="--", label="Background (PoLoN)")
    axes[0].fill_between(x_plot, results["lower_bounds"], results["upper_bounds"],
                         color="blue", alpha=0.15, label="95% CI")
    axes[0].set_title("PoLoN-SB Mean Prediction")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Predicted Counts")
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    axes[1].scatter(X_data, t_data, label="Data", color="orange", alpha=0.7)
    axes[1].scatter(x_plot, results["most_probable_output"], label="Most Probable Output",
                    color="blue")
    axes[1].set_title("PoLoN-SB Most Probable Output")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Most Probable Output")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    plt.subplots_adjust(wspace=0.15, bottom=0.15)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def predict_signal_background_with_plot(X_bg, t_bg, X_signal, t_signal,
                                        bounds_theta=None, theta_init=None,
                                        bounds_signal=None, signal_params0_scaled=None,
                                        X_input=None, n_samples=5000, plot=True,
                                        theta=None, signal_params=None):
    """Predict, then plot (1D inputs). Returns the results dict."""
    results = predict_signal_background(
        X_bg, t_bg, X_signal, t_signal, bounds_theta=bounds_theta, theta_init=theta_init,
        bounds_signal=bounds_signal, signal_params0_scaled=signal_params0_scaled,
        X_input=X_input, n_samples=n_samples, theta=theta, signal_params=signal_params)
    if plot and (np.ndim(results["X_input"]) == 1 or results["X_input"].shape[1] == 1):
        plot_signal_background(results)
    return results
