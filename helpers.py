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

def predict(X_new, X, T, Nv, theta, c_reg=1e-5):
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
    lam_star, H = newton_optimization(T, Nv, C)

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
