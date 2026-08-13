"""
Nonlinear Conjugate Gradient
0, Steepest descent (SD) 
1, Newton method uses the exact Hessian and currently takes a full step alpha=1.
2,Implemented beta choices
FR       : Eq. (5.41a)
PR / PR+ : Eq. (5.44) / Eq. (5.45)
HS       : Eq. (5.46)
FR-PR    : Eq. (5.48)
DY       : Eq. (5.49)
HZ       : Eq. (5.50)

2, Restart
Restart is based on Eq. (5.52):
    |g_k^T g_{k-1}| / ||g_k||^2 >= nu
with the typical value nu = 0.1 stated in the book.

3, Line search (SD and nonlinear CG only; Newton takes a full step)
The step length satisfies the strong Wolfe conditions (5.43):
    armijo:
    f(x_k + alpha_k p_k)
        <= f(x_k) + c1 alpha_k g_k^T p_k                 (5.43a)

    |grad f(x_k + alpha_k p_k)^T p_k|
        <= -c2 g_k^T p_k,                                (5.43b)
 0 < c1 < c2 < 1/2.
c1 = 1e-4, c2 = 0.1 are used in the book (NUMERICAL PERFORMANCE section) and here.
large-scale rosenbrock problem is same as GENROS problem in the book.

Plots
-----
when problem dimension is 2, the following figures are saved for each method:
For each method, the following are saved as THREE SEPARATE figures:
    1. contour
    2. filled topographic/terrain
    3. 3D surface

At each iterate, the figures display
    - the optimization path,
    - the negative gradient -g_k,
    - the actual search direction p_k.

Two figures compare the convergence histories of all methods:
1. objective value versus iteration (convergence_comparison.png)
2. first-order residual versus iteration(residual_norm_vs_iteration.png)
    The vertical axis is the scaled stationarity measure
        ||grad f(x_k)||_inf / (1 + |f(x_k)|),
    shown on a logarithmic scale. The dashed horizontal line marks the
    stopping tolerance from Table 5.1 in the book.

"""
import csv
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import line_search

# ============================================================
# 1. SETTINGS
# ============================================================

# ------------------------------------------------------------
#   "rosenbrock_2d"    : original 2-D teaching example; 
#   "rosenbrock_large" : high-dimensional Rosenbrock benchmark; 
# ------------------------------------------------------------
EXPERIMENT_PROBLEM = "rosenbrock_large"

LARGE_PROBLEM_DIM = 100
#METHODS = ["Newton", "Newton-classic"]
METHODS = ["SD", "Newton-classic", "FR", "PR+", "HS", "FR-PR"]
MAX_ITER = 1000
# Book/Table 5.1 stopping test:
#     ||grad f(x_k)||_inf < GRAD_TOL * (1 + |f(x_k)|).
GRAD_TOL = 1e-5
# Additional safeguard against false convergence after objective explosion.
ABSOLUTE_GRAD_TOL = 1e-5


def make_initial_point(problem_name):

    if problem_name == "rosenbrock_2d":
        return np.array([-1.2, 1.0], dtype=float)

    if problem_name == "rosenbrock_large":
        x0 = np.zeros(LARGE_PROBLEM_DIM, dtype=float)
        x0[1::2] = 0.0
        return x0
    #     return -1.2 * np.ones(
    #     LARGE_PROBLEM_DIM,
    #     dtype=float,
    # )


X0 = make_initial_point(EXPERIMENT_PROBLEM)
PROBLEM_DIM = len(X0)
# Strong Wolfe parameters:
# 0 < c1 < c2 < 1/2
WOLFE_C1 = 1e-4
WOLFE_C2 = 0.1
LINE_SEARCH_MAXITER = 200

# Restart rule Eq.(5.52)
USE_RESTART_552 = True
RESTART_NU = 0.1

# Plot settings
SHOW_VECTOR_ARROWS = True
SHOW_POINT_LABELS = True
MAX_VECTOR_LOCATIONS = 10
MAX_POINT_LABELS = 6
MIN_VECTOR_SPACING = 0.16
MIN_LABEL_SPACING = 0.13


PATH_COLOR = "#3B4CC0"
NEG_GRAD_COLOR = "#D55E00"       # orange-red: negative gradient -g_k
SEARCH_DIR_COLOR = "#009E73"     # bluish green: search direction p_k
NEG_GRAD_ARROW_LENGTH = 0.20
CG_DIR_ARROW_LENGTH = 0.30
ARROW_3D_SCALE = 1.55
ARROW_3D_Z_LIFT = 24.0

X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-1.0, 3.0)
GRID_SIZE = 350
Z_DISPLAY_CAP = 800

SAVE_PLOTS = True
SHOW_PLOTS = False

EXPERIMENT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nonlinear_cg_results",
    f"{EXPERIMENT_PROBLEM}_n{PROBLEM_DIM}",
)
OUTPUT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "figures")
SHOW_CONVERGENCE_COMPARISON = True
SHOW_RESIDUAL_CONVERGENCE = True
SHOW_DIRECTION_COSINES = False
SHOW_NEWTON_MODEL_AGREEMENT = True
SHOW_RUNTIME_COMPARISON = True

SAVE_ITERATION_CSV = True
SAVE_SUMMARY_CSV = True
PRINT_ITERATION_TABLES = False
TABLE_OUTPUT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "tables")


# ============================================================
# 2. Rosenbrock problem
# ============================================================

def rosenbrock(x):
    """
    n-dimensional Rosenbrock function
        f(x) = sum_{i=1}^{n-1} [
                   (1 - x_i)^2
                   + 100 (x_{i+1} - x_i^2)^2
               ].
    Global minimizer: x* = (1, ..., 1), f(x*) = 0.
    """
    x = np.asarray(x, dtype=float)
    d = x[1:] - x[:-1] ** 2
    return float(np.sum((1.0 - x[:-1]) ** 2 + 100.0 * d ** 2))


def rosenbrock_grad(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    d = x[1:] - x[:-1] ** 2
    g[:-1] += -2.0 * (1.0 - x[:-1]) - 400.0 * x[:-1] * d
    g[1:] += 200.0 * d
    return g


def rosenbrock_hessian(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    diag_left = 2.0 - 400.0 * x[1:] + 1200.0 * x[:-1] ** 2
    H[np.arange(n - 1), np.arange(n - 1)] += diag_left
    H[np.arange(1, n), np.arange(1, n)] += 200.0
    off_diag = -400.0 * x[:-1]
    idx = np.arange(n - 1)
    H[idx, idx + 1] = off_diag
    H[idx + 1, idx] = off_diag
    return H


# ============================================================
# 3. Strong-Wolfe line search — Eq. (5.43)
# ============================================================

def strong_wolfe(x, p, g, alpha, c1=WOLFE_C1, c2=WOLFE_C2):
    """
    (5.43a)#avoid step sizes too large
        f(x + alpha p) <= f(x) + c1 alpha g^T p
    (5.43b)#avoid step sizes too small
        |grad f(x + alpha p)^T p| <= -c2 g^T p
    """
    gtp = float(np.dot(g, p))
    if gtp >= 0:
        return False
    x_new = x + alpha * p
    armijo_ok = (
        rosenbrock(x_new)
        <= rosenbrock(x) + c1 * alpha * gtp
    )
    curvature_ok = (
        abs(np.dot(rosenbrock_grad(x_new), p))
        <= -c2 * gtp
    )
    return bool(armijo_ok and curvature_ok)


def choose_step_size(
    x,
    p,
    g,
    c1=WOLFE_C1,
    c2=WOLFE_C2,
    maxiter=LINE_SEARCH_MAXITER,
):
    alpha, _, _, _, _, _ = line_search(
        rosenbrock,
        rosenbrock_grad,
        x,
        p,
        gfk=g,
        c1=c1,
        c2=c2,
        maxiter=maxiter,
    )
    if alpha is None or not strong_wolfe(x, p, g, alpha, c1=c1, c2=c2):
        raise RuntimeError(
            "Strong-Wolfe line search failed to produce a step satisfying "
            "the strong Wolfe conditions."
        )

    return float(alpha)


# ============================================================
# 4. beta update
# ============================================================
def _safe_ratio(numerator, denominator, eps=1e-50):
    if abs(denominator) <= eps:
        return 0.0
    return float(numerator / denominator)


def stopping_measure(f_value, gradient):
    """Return ||grad f||_inf / (1 + |f|), as used in Table 5.1."""
    return float(np.linalg.norm(gradient, ord=np.inf) / (1.0 + abs(f_value)))


def stopping_test(
    f_value,
    gradient,
    tolerance=GRAD_TOL,
    absolute_tolerance=None,
):
    relative_test = stopping_measure(f_value, gradient) < tolerance
    if absolute_tolerance is None:
        return relative_test
    return bool(
        relative_test
        and np.linalg.norm(gradient, ord=np.inf) < absolute_tolerance
    )


def direction_cosine(gradient, direction):
    """Cosine of the angle between the search direction and -gradient."""
    denominator = np.linalg.norm(gradient) * np.linalg.norm(direction)
    if denominator == 0.0:
        return np.nan
    cosine = -float(np.dot(gradient, direction)) / denominator
    return float(np.clip(cosine, -1.0, 1.0))

def beta_fletcher_reeves(g_new, g_old):
    """
    Eq. (5.41a):
        beta_{k+1}^{FR}
            = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)
    """
    return _safe_ratio(
        np.dot(g_new, g_new),
        np.dot(g_old, g_old),
    )


def beta_polak_ribiere(g_new, g_old):
    """
    Eq. (5.44):
        beta_{k+1}^{PR}
            = g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2
    """
    y_hat = g_new - g_old
    return _safe_ratio(
        np.dot(g_new, y_hat),
        np.dot(g_old, g_old),
    )


def beta_pr_plus(g_new, g_old):
    """
    Eq. (5.45):
        beta_{k+1}^{+} = max{beta_{k+1}^{PR}, 0}
    """
    return max(beta_polak_ribiere(g_new, g_old), 0.0)


def beta_hestenes_stiefel(g_new, g_old, p_old):
    """
    Eq. (5.46):
        beta_{k+1}^{HS}
          = g_{k+1}^T (g_{k+1} - g_k)/ [(g_{k+1} - g_k)^T p_k]
    """
    y_hat = g_new - g_old
    return _safe_ratio(
        np.dot(g_new, y_hat),
        np.dot(y_hat, p_old),
    )


def beta_fr_pr(g_new, g_old, next_direction_index):
    """
    Eq. (5.48).
    for k >= 2,
                    -beta_k^{FR},  if beta_k^{PR} < -beta_k^{FR}
        beta_k  =    beta_k^{PR},  if |beta_k^{PR}| <= beta_k^{FR}
                     beta_k^{FR},  if beta_k^{PR} >  beta_k^{FR}.
    """
    beta_pr = beta_polak_ribiere(g_new, g_old)
    if next_direction_index < 2:
        return beta_pr
    beta_fr = beta_fletcher_reeves(g_new, g_old)
    if beta_pr < -beta_fr:
        return -beta_fr
    if beta_pr > beta_fr:
        return beta_fr
    return beta_pr


def beta_eq_549(g_new, g_old, p_old):
    """
    Eq. (5.49):
        beta_{k+1}
          = ||g_{k+1}||^2/ [(g_{k+1} - g_k)^T p_k]
    """
    y_hat = g_new - g_old
    return _safe_ratio(
        np.dot(g_new, g_new),
        np.dot(y_hat, p_old),
    )


def beta_eq_550(g_new, g_old, p_old):
    """
    Eq. (5.50), with y_hat_k = g_{k+1} - g_k:
        beta_{k+1}
          = ( y_hat_k
              - 2 p_k ||y_hat_k||^2 / (y_hat_k^T p_k) )^T
            [ g_{k+1} / (y_hat_k^T p_k) ].
    """
    y_hat = g_new - g_old
    ytp = float(np.dot(y_hat, p_old))
    if abs(ytp) <= 1e-50:
        return 0.0
    vector = y_hat - 2.0 * p_old * (np.dot(y_hat, y_hat) / ytp)
    return float(np.dot(vector, g_new) / ytp)


def compute_beta(method, g_new, g_old, p_old, next_direction_index):
    method = method.upper()
    if method == "SD":
        return 0.0
    if method == "FR":
        return beta_fletcher_reeves(g_new, g_old)
    if method == "PR":
        return beta_polak_ribiere(g_new, g_old)
    if method == "PR+":
        return beta_pr_plus(g_new, g_old)
    if method == "HS":
        return beta_hestenes_stiefel(g_new, g_old, p_old)
    if method == "FR-PR":
        return beta_fr_pr(
            g_new, g_old,
            next_direction_index=next_direction_index,
        )
    if method == "5.49":
        return beta_eq_549(g_new, g_old, p_old)
    if method == "5.50":
        return beta_eq_550(g_new, g_old, p_old)

    raise ValueError(
        "method must be one of "
        "{'SD', 'FR', 'PR', 'PR+', 'HS', 'FR-PR', '5.49', '5.50'}"
    )


# ============================================================
# 5. Restart — Eq. (5.52)
# ============================================================

def restart_test_552(g_k, g_k_minus_1, nu=RESTART_NU):
    """
    Eq. (5.52):
        |g_k^T g_{k-1}| / ||g_k||^2 >= nu.
    The book gives nu = 0.1 as a typical value.
    If the test is true:
        p_k = -g_k,
        beta_k = 0.
    """
    denominator = np.dot(g_k, g_k)
    if denominator <= 1e-50:
        return False, 0.0
    measure = abs(np.dot(g_k, g_k_minus_1)) / denominator
    return bool(measure >= nu), float(measure)


# ============================================================
# 6. Steepest descent, nonlinear conjugate-gradient, and Newton
# ============================================================

class OptimizationFailure(RuntimeError):
    """A classified failure of one solver run."""

    def __init__(self, status, method, iteration, message):
        super().__init__(message)
        self.status = status
        self.method = method
        self.iteration = iteration


def nonlinear_cg(
    x0,
    method="FR",
    max_iter=100,
    grad_tol=1e-5,
    use_restart_552=True,
    restart_nu=0.1,
    absolute_grad_tol=None,
    wolfe_c1=WOLFE_C1,
    wolfe_c2=WOLFE_C2,
    line_search_maxiter=LINE_SEARCH_MAXITER,
    print_failure_diagnostics=True,
):
    """
        p_0 = -g_0
        x_{k+1} = x_k + alpha_k p_k
        p_{k+1} = -g_{k+1} + beta_{k+1} p_k.                (5.41b)
    alpha_k is required to satisfy the strong Wolfe conditions (5.43).
    Restart uses Eq. (5.52)
    """
    start_time = time.perf_counter()
    line_search_time = 0.0
    x = np.asarray(x0, dtype=float).copy()
    g = rosenbrock_grad(x)
    p = -g.copy()
    iterates = [x.copy()]
    gradients = []
    directions = []
    direction_cosines = []
    alphas = []
    betas = []
    restarted = []
    restart_measures = []
    f_values = [rosenbrock(x)]
    for k in range(max_iter):
        if stopping_test(
            f_values[-1], g, grad_tol, absolute_grad_tol
        ):
            break

        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0:
            raise OptimizationFailure(
                "non_descent_direction",
                method,
                k,
                f"At iteration k={k}, g_k^T p_k = {gtp:.6e} >= 0, "
                "so p_k is not a finite descent direction. ",
            )
        gradients.append(g.copy())
        directions.append(p.copy())
        direction_cosines.append(direction_cosine(g, p))

        ls_start = time.perf_counter()
        try:
            alpha = choose_step_size(
                x,
                p,
                g,
                c1=wolfe_c1,
                c2=wolfe_c2,
                maxiter=line_search_maxiter,
            )
        except RuntimeError as exc:
            line_search_time += time.perf_counter() - ls_start
            previous_beta = betas[-1] if betas else np.nan
            previous_restart = restarted[-1] if restarted else False
            previous_restart_measure = (
                restart_measures[-1] if restart_measures else np.nan
            )
            if print_failure_diagnostics:
                print("\n[LINE-SEARCH DEBUG]")
                print(f"  method                   = {method}")
                print(f"  iteration k              = {k}")
                print(f"  f(x_k)                   = {rosenbrock(x):.16e}")
                print(f"  ||g_k||_2                = {np.linalg.norm(g):.16e}")
                print(f"  ||p_k||_2                = {np.linalg.norm(p):.16e}")
                print(f"  g_k^T p_k                = {gtp:.16e}")
                print(
                    "  cos(angle(p_k, -g_k))    = "
                    f"{direction_cosines[-1]:.16e}"
                )
                print(f"  beta used in p_k         = {previous_beta:.16e}")
                print(
                    "  previous restart measure = "
                    f"{previous_restart_measure:.16e}"
                )
                print(f"  previous restart         = {previous_restart}")
            raise OptimizationFailure(
                "line_search_failed",
                method,
                k,
                f"{method} line search failed at iteration k={k}",
            ) from exc
        else:
            line_search_time += time.perf_counter() - ls_start
        alphas.append(alpha)


        x_new = x + alpha * p
        g_new = rosenbrock_grad(x_new)
        f_new = rosenbrock(x_new)
        if not np.isfinite(f_new) or not np.all(np.isfinite(g_new)):
            raise OptimizationFailure(
                "numerical_failure",
                method,
                k,
                f"{method}: non-finite objective or gradient after "
                f"iteration {k}",
            )

        iterates.append(x_new.copy())
        f_values.append(f_new)


        did_restart = False
        restart_measure = 0.0

        if method == "SD":
            beta = 0.0
        else:
            beta = compute_beta(
                method,
                g_new=g_new,
                g_old=g,
                p_old=p,
                next_direction_index=k + 1,
            )

            if use_restart_552:
                did_restart, restart_measure = restart_test_552(
                    g_k=g_new,
                    g_k_minus_1=g,
                    nu=restart_nu,
                )
                if did_restart:
                    beta = 0.0


        p_new = -g_new + beta * p

        betas.append(beta)
        restarted.append(did_restart)
        restart_measures.append(restart_measure)

        x = x_new
        g = g_new
        p = p_new

    final_g = rosenbrock_grad(x)
    final_f = rosenbrock(x)
    return {
        "method": method,
        "x0": np.asarray(x0, dtype=float),
        "iterates": np.asarray(iterates),
        "gradients": np.asarray(gradients),
        "directions": np.asarray(directions),
        "direction_cosines": np.asarray(direction_cosines),
        "alphas": np.asarray(alphas),
        "betas": np.asarray(betas),
        "restarted": np.asarray(restarted, dtype=bool),
        "restart_measures": np.asarray(restart_measures),
        "f_values": np.asarray(f_values),
        "final_x": x.copy(),
        "final_f": final_f,
        "final_grad_norm": np.linalg.norm(final_g),
        "final_grad_inf_norm": np.linalg.norm(final_g, ord=np.inf),
        "final_stopping_measure": stopping_measure(final_f, final_g),
        "converged": stopping_test(
            final_f, final_g, grad_tol, absolute_grad_tol
        ),
        "num_steps": len(alphas),
        "elapsed_time": time.perf_counter() - start_time,
        "line_search_time": line_search_time,
        "hessian_build_time": 0.0,
        "hessian_check_time": 0.0,
        "linear_solve_time": 0.0,
    }


def newton_method(
    x0,
    max_iter=100,
    grad_tol=1e-5,
    absolute_grad_tol=None,
    min_eigenvalue=1e-8,
):
    start_time = time.perf_counter()
    hessian_build_time = 0.0
    hessian_check_time = 0.0
    linear_solve_time = 0.0
    x = np.asarray(x0, dtype=float).copy()
    iterates = [x.copy()]
    gradients = []
    directions = []
    direction_cosines = []
    hessian_shifts = []
    hessian_min_eigenvalues = []
    predicted_reductions = []
    actual_reductions = []
    model_agreement_ratios = []
    f_values = [rosenbrock(x)]

    for k in range(max_iter):
        g = rosenbrock_grad(x)
        if stopping_test(
            f_values[-1], g, grad_tol, absolute_grad_tol
        ):
            break
        hess_start = time.perf_counter()
        hessian = rosenbrock_hessian(x)
        hessian_build_time += time.perf_counter() - hess_start
        check_start = time.perf_counter()
        smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
        hessian_check_time += time.perf_counter() - check_start
        shift = max(0.0, min_eigenvalue - smallest_eigenvalue)
        try:
            solve_start = time.perf_counter()
            p = np.linalg.solve(hessian + shift * np.eye(len(x)), -g)
            linear_solve_time += time.perf_counter() - solve_start
        except np.linalg.LinAlgError as exc:
            raise OptimizationFailure(
                "linear_solve_failed",
                "Newton",
                k,
                f"Modified Newton linear solve failed at iteration {k}",
            ) from exc

        gradients.append(g.copy())
        directions.append(p.copy())
        direction_cosines.append(direction_cosine(g, p))
        hessian_shifts.append(shift)
        hessian_min_eigenvalues.append(smallest_eigenvalue)
        step = p
        predicted_reduction = -float(np.dot(g, step) + 0.5 * np.dot(step, hessian @ step))
        if not np.isfinite(predicted_reduction):
            raise OptimizationFailure(
                "numerical_failure",
                "Newton",
                k,
                f"Non-finite predicted reduction at iteration {k}",
            )
        f_old = f_values[-1]
        x = x + step
        f_new = rosenbrock(x)
        if not np.isfinite(f_new) or not np.all(np.isfinite(x)):
            raise OptimizationFailure(
                "numerical_failure",
                "Newton",
                k,
                f"Newton: non-finite iterate or objective after iteration {k}",
            )
        actual_reduction = f_old - f_new
        if not np.isfinite(actual_reduction):
            raise OptimizationFailure(
                "numerical_failure",
                "Newton",
                k,
                f"Non-finite actual reduction at iteration {k}",
            )
        if predicted_reduction == 0.0 or not np.isfinite(predicted_reduction):
            rho = np.nan
        else:
            rho = actual_reduction / predicted_reduction
        predicted_reductions.append(predicted_reduction)
        actual_reductions.append(actual_reduction)
        model_agreement_ratios.append(rho)
        iterates.append(x.copy())
        f_values.append(f_new)

    num_steps = len(directions)
    final_g = rosenbrock_grad(x)
    final_f = rosenbrock(x)
    return {
        "method": "Newton",
        "x0": np.asarray(x0, dtype=float),
        "iterates": np.asarray(iterates),
        "gradients": np.asarray(gradients),
        "directions": np.asarray(directions),
        "direction_cosines": np.asarray(direction_cosines),
        "betas": np.full(num_steps, np.nan),
        "restarted": np.zeros(num_steps, dtype=bool),
        "restart_measures": np.full(num_steps, np.nan),
        "hessian_shifts": np.asarray(hessian_shifts),
        "hessian_min_eigenvalues": np.asarray(hessian_min_eigenvalues),
        "predicted_reductions": np.asarray(predicted_reductions),
        "actual_reductions": np.asarray(actual_reductions),
        "model_agreement_ratios": np.asarray(model_agreement_ratios),
        "f_values": np.asarray(f_values),
        "final_x": x.copy(),
        "final_f": final_f,
        "final_grad_norm": np.linalg.norm(final_g),
        "final_grad_inf_norm": np.linalg.norm(final_g, ord=np.inf),
        "final_stopping_measure": stopping_measure(final_f, final_g),
        "converged": stopping_test(
            final_f, final_g, grad_tol, absolute_grad_tol
        ),
        "num_steps": num_steps,
        "elapsed_time": time.perf_counter() - start_time,
        "hessian_build_time": hessian_build_time,
        "hessian_check_time": hessian_check_time,
        "linear_solve_time": linear_solve_time,
    }


def classical_newton_method(
    x0,
    max_iter=100,
    grad_tol=1e-5,
    absolute_grad_tol=None,
):
    method = "Newton-classic"
    start_time = time.perf_counter()
    hessian_build_time = 0.0
    linear_solve_time = 0.0
    x = np.asarray(x0, dtype=float).copy()
    iterates = [x.copy()]
    gradients = []
    directions = []
    direction_cosines = []
    predicted_reductions = []
    actual_reductions = []
    model_agreement_ratios = []
    f_values = [rosenbrock(x)]

    for k in range(max_iter):
        g = rosenbrock_grad(x)
        if stopping_test(
            f_values[-1], g, grad_tol, absolute_grad_tol
        ):
            break

        build_start = time.perf_counter()
        hessian = rosenbrock_hessian(x)
        hessian_build_time += time.perf_counter() - build_start
        try:
            solve_start = time.perf_counter()
            p = np.linalg.solve(hessian, -g)
            linear_solve_time += time.perf_counter() - solve_start
        except np.linalg.LinAlgError as exc:
            raise OptimizationFailure(
                "linear_solve_failed",
                method,
                k,
                f"Classical Newton linear solve failed at iteration {k}",
            ) from exc

        if not np.all(np.isfinite(p)):
            raise OptimizationFailure(
                "numerical_failure",
                method,
                k,
                f"Classical Newton direction is non-finite at iteration {k}",
            )

        gradients.append(g.copy())
        directions.append(p.copy())
        direction_cosines.append(direction_cosine(g, p))

        with np.errstate(over="ignore", invalid="ignore"):
            predicted_reduction = -float(
                np.dot(g, p) + 0.5 * np.dot(p, hessian @ p)
            )
        f_old = f_values[-1]
        x = x + p
        f_new = rosenbrock(x)
        if not np.all(np.isfinite(x)) or not np.isfinite(f_new):
            raise OptimizationFailure(
                "numerical_failure",
                method,
                k,
                f"Classical Newton produced a non-finite iterate or "
                f"objective at iteration {k}",
            )

        actual_reduction = f_old - f_new
        rho = (
            actual_reduction / predicted_reduction
            if np.isfinite(predicted_reduction)
            and predicted_reduction != 0.0
            else np.nan
        )
        predicted_reductions.append(predicted_reduction)
        actual_reductions.append(actual_reduction)
        model_agreement_ratios.append(rho)
        iterates.append(x.copy())
        f_values.append(f_new)

    num_steps = len(directions)
    final_g = rosenbrock_grad(x)
    final_f = rosenbrock(x)
    return {
        "method": method,
        "x0": np.asarray(x0, dtype=float),
        "iterates": np.asarray(iterates),
        "gradients": np.asarray(gradients),
        "directions": np.asarray(directions),
        "direction_cosines": np.asarray(direction_cosines),
        "betas": np.full(num_steps, np.nan),
        "restarted": np.zeros(num_steps, dtype=bool),
        "restart_measures": np.full(num_steps, np.nan),
        "predicted_reductions": np.asarray(predicted_reductions),
        "actual_reductions": np.asarray(actual_reductions),
        "model_agreement_ratios": np.asarray(model_agreement_ratios),
        "f_values": np.asarray(f_values),
        "final_x": x.copy(),
        "final_f": final_f,
        "final_grad_norm": np.linalg.norm(final_g),
        "final_grad_inf_norm": np.linalg.norm(final_g, ord=np.inf),
        "final_stopping_measure": stopping_measure(final_f, final_g),
        "converged": stopping_test(
            final_f, final_g, grad_tol, absolute_grad_tol
        ),
        "num_steps": num_steps,
        "elapsed_time": time.perf_counter() - start_time,
        "hessian_build_time": hessian_build_time,
        "hessian_check_time": 0.0,
        "linear_solve_time": linear_solve_time,
    }


def is_newton_method(method):
    return method in {"Newton", "Newton-classic"}


# ============================================================
# 7. Plotting
# ============================================================

def normalized_arrow(v, length):
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return length * v / norm


def representative_indices(count, max_count):
    if count <= max_count:
        return np.arange(count, dtype=int)

    early_count = min(3, max_count)
    early = np.arange(early_count, dtype=int)
    later = np.linspace(
        early_count,
        count - 1,
        max_count - early_count,
        dtype=int,
    )
    return np.unique(np.concatenate((early, later)))


def spatially_separated_indices(points, max_count, min_spacing):
    points = np.asarray(points)
    if len(points) == 0:
        return np.array([], dtype=int)

    selected = [0]
    for k in range(1, len(points)):
        distances = np.linalg.norm(points[selected] - points[k], axis=1)
        if np.all(distances >= min_spacing):
            selected.append(k)


    last = len(points) - 1
    if last not in selected:
        distances = np.linalg.norm(points[selected] - points[last], axis=1)
        if np.all(distances >= 0.65 * min_spacing):
            selected.append(last)

    if len(selected) > max_count:
        keep = representative_indices(len(selected), max_count)
        selected = [selected[i] for i in keep]

    return np.asarray(selected, dtype=int)


def add_direction_legend(ax, result, loc="upper left"):
    """Add consistent proxy artists for arrows in both 2D and 3D plots."""
    if not SHOW_VECTOR_ARROWS:
        return

    handles, labels = ax.get_legend_handles_labels()
    if result["method"] == "SD":
        handles.append(Line2D(
            [0], [0], color=SEARCH_DIR_COLOR, linewidth=2.8,
            marker=">", markersize=7,
        ))
        labels.append(r"$p_k=-g_k$ (SD)")
    else:
        handles.extend([
            Line2D(
                [0], [0], color=NEG_GRAD_COLOR, linewidth=2.2,
                marker=">", markersize=6,
            ),
            Line2D(
                [0], [0], color=SEARCH_DIR_COLOR, linewidth=3.0,
                marker=">", markersize=7,
            ),
        ])
        direction_label = (
            r"Newton direction $p_k$"
            if result["method"] == "Newton"
            else r"search direction $p_k$"
        )
        labels.extend([r"negative gradient $-g_k$", direction_label])

    ax.legend(handles, labels, loc=loc, framealpha=0.9)


def make_grid():
    x1 = np.linspace(X1_RANGE[0], X1_RANGE[1], GRID_SIZE)
    x2 = np.linspace(X2_RANGE[0], X2_RANGE[1], GRID_SIZE)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1.0 - X1) ** 2 + 100.0 * (X2 - X1 ** 2) ** 2
    return X1, X2, Z


def add_2d_vectors(ax, result):
    if not SHOW_VECTOR_ARROWS:
        return
    xs = result["iterates"]
    gs = result["gradients"]
    ps = result["directions"]

    indices = spatially_separated_indices(
        xs[:result["num_steps"]],
        MAX_VECTOR_LOCATIONS,
        MIN_VECTOR_SPACING,
    )

    for k in indices:
        xk = xs[k]


        pp = normalized_arrow(ps[k], CG_DIR_ARROW_LENGTH)
        ax.quiver(
            xk[0], xk[1],
            pp[0], pp[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=SEARCH_DIR_COLOR,
            width=0.007,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            alpha=0.88,
            zorder=5,
        )


        if result["method"] != "SD":
            ng = normalized_arrow(-gs[k], NEG_GRAD_ARROW_LENGTH)
            ax.quiver(
                xk[0], xk[1],
                ng[0], ng[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=NEG_GRAD_COLOR,
                width=0.004,
                headwidth=4.5,
                headlength=6.0,
                headaxislength=5.0,
                alpha=0.95,
                zorder=6,
            )


def add_point_labels_2d(ax, result):
    if not SHOW_POINT_LABELS:
        return

    xs = result["iterates"]
    label_indices = spatially_separated_indices(
        xs,
        MAX_POINT_LABELS,
        MIN_LABEL_SPACING,
    )
    for k in label_indices:
        x = xs[k]
        ax.text(
            x[0] + 0.025,
            x[1] + 0.025,
            f"$x_{k}$",
            fontsize=8,
            color="#222222",
            zorder=7,
        )


def add_3d_vectors(ax, result):
    if not SHOW_VECTOR_ARROWS:
        return

    xs = result["iterates"]
    gs = result["gradients"]
    ps = result["directions"]

    indices = spatially_separated_indices(
        xs[:result["num_steps"]],
        MAX_VECTOR_LOCATIONS,
        MIN_VECTOR_SPACING,
    )

    for k in indices:
        xk = xs[k]

        zk = min(rosenbrock(xk) + ARROW_3D_Z_LIFT, Z_DISPLAY_CAP)

        pp = normalized_arrow(
            ps[k],
            ARROW_3D_SCALE * CG_DIR_ARROW_LENGTH,
        )
        ax.quiver(
            xk[0], xk[1], zk,
            pp[0], pp[1], 0.0,
            length=1.0,
            normalize=False,
            color=SEARCH_DIR_COLOR,
            linewidth=3.0,
            arrow_length_ratio=0.28,
            alpha=0.95,
        )

        if result["method"] != "SD":
            ng = normalized_arrow(
                -gs[k],
                ARROW_3D_SCALE * NEG_GRAD_ARROW_LENGTH,
            )
            ax.quiver(
                xk[0], xk[1], min(zk + 4.0, Z_DISPLAY_CAP),
                ng[0], ng[1], 0.0,
                length=1.0,
                normalize=False,
                color=NEG_GRAD_COLOR,
                linewidth=2.2,
                arrow_length_ratio=0.30,
                alpha=1.0,
            )


def _save_or_show(fig, filename):
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=220,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def plot_contour(result):
    X1, X2, Z = make_grid()
    xs = result["iterates"]
    method = result["method"]

    fig, ax = plt.subplots(figsize=(8.0, 6.6))

    contour_levels = np.logspace(-1, 3.2, 22)
    ax.contour(X1, X2, Z, levels=contour_levels, linewidths=0.9)

    ax.plot(
        xs[:, 0], xs[:, 1],
        "-o",
        color=PATH_COLOR,
        linewidth=2.2,
        markersize=4.5,
        label=f"{method} path",
    )
    ax.scatter(
        xs[0, 0], xs[0, 1], s=110, marker="o",
        label="$x_0$", zorder=8,
    )
    ax.scatter(
        1.0, 1.0, s=150, marker="*",
        label="$x^*=(1,1)$", zorder=9,
    )

    add_2d_vectors(ax, result)
    add_point_labels_2d(ax, result)

    ax.set_xlim(X1_RANGE)
    ax.set_ylim(X2_RANGE)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Optimization on Rosenbrock — {method} — Contour")
    ax.grid(alpha=0.25)
    add_direction_legend(ax, result, loc="upper right")

    fig.tight_layout()
    _save_or_show(fig, f"{method}_contour.png")


def plot_terrain(result):
    X1, X2, Z = make_grid()
    xs = result["iterates"]
    method = result["method"]

    fig, ax = plt.subplots(figsize=(8.0, 6.6))

    terrain_field = np.log10(1.0 + Z)
    cf = ax.contourf(
        X1, X2,
        terrain_field,
        levels=45,
        cmap="viridis",
    )
    ax.contour(
        X1, X2,
        terrain_field,
        levels=18,
        linewidths=0.45,
        alpha=0.5,
    )

    ax.plot(
        xs[:, 0], xs[:, 1],
        "-o",
        color=PATH_COLOR,
        linewidth=2.3,
        markersize=4.5,
        label=f"{method} path",
    )
    ax.scatter(xs[0, 0], xs[0, 1], s=110, zorder=8)
    ax.scatter(1.0, 1.0, s=150, marker="*", zorder=9)

    add_2d_vectors(ax, result)
    add_point_labels_2d(ax, result)

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(1+f(x))$")

    ax.set_xlim(X1_RANGE)
    ax.set_ylim(X2_RANGE)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Optimization on Rosenbrock — {method} — Terrain")
    add_direction_legend(ax, result, loc="upper left")

    fig.tight_layout()
    _save_or_show(fig, f"{method}_terrain.png")


def plot_surface_3d(result):
    X1, X2, Z = make_grid()
    Z3 = np.minimum(Z, Z_DISPLAY_CAP)

    xs = result["iterates"]
    path_z = np.minimum(
        np.array([rosenbrock(x) for x in xs]),
        Z_DISPLAY_CAP,
    )
    method = result["method"]

    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    stride = max(1, GRID_SIZE // 90)

    ax.plot_surface(
        X1[::stride, ::stride],
        X2[::stride, ::stride],
        Z3[::stride, ::stride],
        cmap="viridis",
        alpha=0.72,
        linewidth=0,
        antialiased=True,
    )

    ax.plot(
        xs[:, 0],
        xs[:, 1],
        path_z,
        "-o",
        color=PATH_COLOR,
        linewidth=2.5,
        markersize=4.5,
        label=f"{method} path",
    )

    ax.scatter(xs[0, 0], xs[0, 1], path_z[0], s=70)
    ax.scatter(1.0, 1.0, 0.0, s=110, marker="*")

    add_3d_vectors(ax, result)

    ax.set_xlim(X1_RANGE)
    ax.set_ylim(X2_RANGE)
    ax.set_zlim(0, Z_DISPLAY_CAP)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    ax.set_title(f"Optimization on Rosenbrock — {method} — 3D Surface")
    ax.view_init(elev=31, azim=-56)
    add_direction_legend(ax, result, loc="upper left")

    fig.tight_layout()
    _save_or_show(fig, f"{method}_surface_3d.png")


def plot_three_separate_views(result):

    plot_contour(result)
    plot_terrain(result)
    plot_surface_3d(result)


# ============================================================
# 8. Iteration data: CSV export and optional terminal table
# ============================================================

def save_iteration_csv(result):
    """Save dimension-independent scalar diagnostics for every iteration."""
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    filename = f"{result['method']}_iterations.csv"
    filepath = os.path.join(TABLE_OUTPUT_DIR, filename)
    fieldnames = [
        "method", "iteration", "f_x", "stopping_measure",
        "grad_norm_2", "grad_norm_inf", "direction_norm_2",
        "direction_cosine", "alpha", "beta_next", "restart_measure",
        "restarted", "hessian_min_eigenvalue", "hessian_shift",
        "predicted_reduction", "actual_reduction",
        "model_agreement_ratio", "is_final",
    ]

    num_steps = result["num_steps"]
    method = result["method"]
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for k, xk in enumerate(result["iterates"]):
            is_final = k == num_steps
            f_k = float(result["f_values"][k])
            gk = (
                rosenbrock_grad(xk)
                if is_final else result["gradients"][k]
            )
            row = {
                "method": method,
                "iteration": k,
                "f_x": f_k,
                "stopping_measure": stopping_measure(f_k, gk),
                "grad_norm_2": float(np.linalg.norm(gk)),
                "grad_norm_inf": float(np.linalg.norm(gk, ord=np.inf)),
                "direction_norm_2": "",
                "direction_cosine": "",
                "alpha": "",
                "beta_next": "",
                "restart_measure": "",
                "restarted": "",
                "hessian_min_eigenvalue": "",
                "hessian_shift": "",
                "predicted_reduction": "",
                "actual_reduction": "",
                "model_agreement_ratio": "",
                "is_final": is_final,
            }

            if not is_final:
                pk = result["directions"][k]
                row["direction_norm_2"] = float(np.linalg.norm(pk))
                row["direction_cosine"] = float(
                    result["direction_cosines"][k]
                )
                if is_newton_method(method):
                    row.update({
                        "predicted_reduction": float(
                            result["predicted_reductions"][k]
                        ),
                        "actual_reduction": float(
                            result["actual_reductions"][k]
                        ),
                        "model_agreement_ratio": float(
                            result["model_agreement_ratios"][k]
                        ),
                    })
                    if method == "Newton":
                        row.update({
                            "hessian_min_eigenvalue": float(
                                result["hessian_min_eigenvalues"][k]
                            ),
                            "hessian_shift": float(
                                result["hessian_shifts"][k]
                            ),
                        })
                else:
                    row.update({
                        "alpha": float(result["alphas"][k]),
                        "beta_next": float(result["betas"][k]),
                        "restart_measure": float(
                            result["restart_measures"][k]
                        ),
                        "restarted": bool(result["restarted"][k]),
                    })
            writer.writerow(row)

    return filepath


def save_summary_csv(results):
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(TABLE_OUTPUT_DIR, "methods_summary.csv")
    fieldnames = [
        "problem", "dimension", "method", "num_steps",
        "final_x_norm", "final_x_error_2", "final_x_error_inf",
        "final_f", "final_grad_norm", "final_grad_inf_norm",
        "final_stopping_measure", "min_direction_cosine",
        "min_model_agreement_ratio", "num_negative_model_agreement",
        "converged",
        "elapsed_time_s", "line_search_time_s",
        "hessian_build_time_s", "hessian_check_time_s",
        "linear_solve_time_s",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for method, result in results.items():
            final_x = np.asarray(result["final_x"], dtype=float)
            final_x_error = final_x - np.ones_like(final_x)
            rho_values = np.asarray(
                result.get("model_agreement_ratios", []), dtype=float
            )
            finite_rho_values = rho_values[np.isfinite(rho_values)]
            writer.writerow({
                "problem": EXPERIMENT_PROBLEM,
                "dimension": len(final_x),
                "method": method,
                "num_steps": result["num_steps"],
                "final_x_norm": float(np.linalg.norm(final_x)),
                "final_x_error_2": float(np.linalg.norm(final_x_error)),
                "final_x_error_inf": float(
                    np.linalg.norm(final_x_error, ord=np.inf)
                ),
                "final_f": float(result["final_f"]),
                "final_grad_norm": float(result["final_grad_norm"]),
                "final_grad_inf_norm": float(result["final_grad_inf_norm"]),
                "final_stopping_measure": float(result["final_stopping_measure"]),
                "min_direction_cosine": (
                    float(np.min(result["direction_cosines"]))
                    if len(result["direction_cosines"]) else np.nan
                ),
                "min_model_agreement_ratio": (
                    float(np.min(finite_rho_values))
                    if len(finite_rho_values) else np.nan
                ),
                "num_negative_model_agreement": int(
                    np.count_nonzero(finite_rho_values < 0.0)
                ),
                "converged": bool(result["converged"]),
                "elapsed_time_s": float(result.get("elapsed_time", np.nan)),
                "line_search_time_s": float(result.get("line_search_time", np.nan)),
                "hessian_build_time_s": float(result.get("hessian_build_time", np.nan)),
                "hessian_check_time_s": float(result.get("hessian_check_time", np.nan)),
                "linear_solve_time_s": float(result.get("linear_solve_time", np.nan)),
            })

    return filepath

def print_iteration_table(result):
    method = result["method"]
    xs = result["iterates"]
    gs = result["gradients"]
    betas = result["betas"]
    restarted = result["restarted"]
    restart_measures = result["restart_measures"]
    fvals = result["f_values"]
    if is_newton_method(method):
        print("\n" + "=" * 82)
        if method == "Newton":
            print("MODIFIED FULL-STEP NEWTON METHOD (NO LINE SEARCH)")
        else:
            print("CLASSICAL FULL-STEP NEWTON METHOD (NO MODIFICATION)")
        print("=" * 82)
        print(
            f"{'k':>3s} | {'x1':>12s} {'x2':>12s} | "
            f"{'f(x_k)':>14s} | {'||g_k||':>12s}"
            + (f" | {'H shift':>11s}" if method == "Newton" else "")
        )
        print("-" * 82)
        for k in range(result["num_steps"]):
            line = (
                f"{k:3d} | {xs[k, 0]:12.6f} {xs[k, 1]:12.6f} | "
                f"{fvals[k]:14.6e} | {np.linalg.norm(gs[k]):12.5e}"
            )
            if method == "Newton":
                line += f" | {result['hessian_shifts'][k]:11.4e}"
            print(line)
        print("-" * 82)
        print(f"Final x = [{result['final_x'][0]:.10f}, {result['final_x'][1]:.10f}]")
        print(f"Final f(x)        = {result['final_f']:.6e}")
        print(f"Final ||grad f||  = {result['final_grad_norm']:.6e}")
        print(f"Number of steps   = {result['num_steps']}")
        return

    alphas = result["alphas"]
    print("\n" + "=" * 133)
    algorithm = "STEEPEST DESCENT" if method == "SD" else "NONLINEAR CG"
    print(f"{algorithm} — {method}")
    if method == "SD":
        print(
            f"Strong Wolfe: c1={WOLFE_C1:g}, c2={WOLFE_C2:g}; "
            "direction p_k = -g_k"
        )
    else:
        print(
            f"Strong Wolfe: c1={WOLFE_C1:g}, c2={WOLFE_C2:g}; "
            f"restart Eq. (5.52): nu={RESTART_NU:g}"
        )
    print("=" * 133)
    print(
        f"{'k':>3s} | {'x1':>12s} {'x2':>12s} | "
        f"{'f(x_k)':>14s} | {'||g_k||':>12s} | "
        f"{'alpha_k':>11s} | {'beta_{k+1}':>13s} | "
        f"{'R_5.52':>11s} | {'restart':>8s}"
    )
    print("-" * 133)
    for k in range(result["num_steps"]):
        xk = xs[k]
        gk = gs[k]
        print(
            f"{k:3d} | "
            f"{xk[0]:12.6f} {xk[1]:12.6f} | "
            f"{fvals[k]:14.6e} | "
            f"{np.linalg.norm(gk):12.5e} | "
            f"{alphas[k]:11.4e} | "
            f"{betas[k]:13.4e} | "
            f"{restart_measures[k]:11.4e} | "
            f"{str(restarted[k]):>8s}"
        )

    print("-" * 133)
    print(
        f"Final x = [{result['final_x'][0]:.10f}, "
        f"{result['final_x'][1]:.10f}]"
    )
    print(f"Final f(x)        = {result['final_f']:.6e}")
    print(f"Final ||grad f||  = {result['final_grad_norm']:.6e}")
    print(f"Number of steps   = {result['num_steps']}")


# ============================================================
# 9. Convergence comparison
# ============================================================

def plot_convergence_comparison(results):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method, result in results.items():
        values = np.maximum(result["f_values"], 1e-16)
        ax.semilogy(
            np.arange(len(values)),
            values,
            "-o",
            markersize=4,
            linewidth=2,
            label=method,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title(f"Optimization methods: convergence — {EXPERIMENT_PROBLEM}, n={PROBLEM_DIM}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_or_show(fig, "convergence_comparison.png")


def plot_residual_convergence(results):
    """Plot the scaled first-order measure used by the stopping test.

    Table 5.1 stops when
        ||grad f(x_k)||_inf / (1 + |f(x_k)|) < GRAD_TOL.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for method, result in results.items():
        residual_measures = np.array([
            stopping_measure(fk, rosenbrock_grad(xk))
            for xk, fk in zip(result["iterates"], result["f_values"])
        ])
        residual_measures = np.maximum(residual_measures, 1e-16)

        ax.semilogy(
            np.arange(len(residual_measures)),
            residual_measures,
            "-o",
            markersize=4,
            linewidth=2,
            label=method,
        )

    # The same tolerance used by the stopping test.
    ax.axhline(
        GRAD_TOL,
        linestyle="--",
        linewidth=1.5,
        label=rf"tolerance $={GRAD_TOL:.0e}$",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\nabla f(x_k)\|_\infty/(1+|f(x_k)|)$")
    ax.set_title(f"Scaled stationarity measure — {EXPERIMENT_PROBLEM}, n={PROBLEM_DIM}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_or_show(fig, "residual_norm_vs_iteration.png")


def plot_direction_cosines(results):
    """Plot cos(theta_k), where theta_k is the angle between p_k and -g_k."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for method, result in results.items():
        cosines = result["direction_cosines"]
        ax.plot(
            np.arange(len(cosines)),
            cosines,
            "-o",
            markersize=4,
            linewidth=2,
            label=method,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\cos\theta_k=-g_k^T p_k/(\|g_k\|_2\|p_k\|_2)$")
    ax.set_title(
        f"Search-direction quality — {EXPERIMENT_PROBLEM}, n={PROBLEM_DIM}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_or_show(fig, "direction_cosine_vs_iteration.png")


def plot_newton_model_agreement(result):
    rho = np.asarray(result["model_agreement_ratios"], dtype=float)
    finite = np.isfinite(rho)
    iterations = np.arange(len(rho))
    positive = finite & (rho >= 0.0)
    negative = finite & (rho < 0.0)

    fig, (ax_rho, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    ax_rho.scatter(
        iterations[positive],
        rho[positive],
        s=20,
        alpha=0.78,
        color="#009E73",
        edgecolors="none",
        label=rf"actual decrease ($\rho_k\geq0$): {np.count_nonzero(positive)}",
    )
    ax_rho.scatter(
        iterations[negative],
        rho[negative],
        s=22,
        alpha=0.82,
        color="#D55E00",
        edgecolors="none",
        label=rf"objective increased ($\rho_k<0$): {np.count_nonzero(negative)}",
    )
    ax_rho.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                   label="perfect model agreement")
    ax_rho.axhline(0.0, color="#7F0000", linestyle=":", linewidth=1.2,
                   label="decrease/increase boundary")
    ax_rho.set_yscale("symlog", linthresh=0.1)
    ax_rho.set_ylabel(
        r"$\rho_k=(f_k-f_{k+1})/[-g_k^T s_k-\frac{1}{2}s_k^T H_k s_k]$"
    )
    ax_rho.set_title(
        f"{result['method']} quadratic-model agreement — "
        f"{EXPERIMENT_PROBLEM}, "
        f"n={PROBLEM_DIM}"
    )
    ax_rho.grid(alpha=0.25)
    ax_rho.legend(loc="best")

    finite_count = np.cumsum(finite.astype(int))
    negative_count = np.cumsum(negative.astype(int))
    cumulative_negative_rate = np.divide(
        negative_count,
        finite_count,
        out=np.full(len(rho), np.nan, dtype=float),
        where=finite_count > 0,
    )
    ax_rate.plot(
        iterations,
        cumulative_negative_rate,
        color="#6A3D9A",
        linewidth=2.0,
        label="cumulative fraction with objective increase",
    )
    if np.any(finite):
        final_rate = cumulative_negative_rate[np.flatnonzero(finite)[-1]]
        ax_rate.axhline(
            final_rate,
            color="#6A3D9A",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax_rate.text(
            0.99,
            final_rate,
            f" final: {final_rate:.1%}",
            transform=ax_rate.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="#6A3D9A",
        )
    ax_rate.set_ylim(-0.02, 1.02)
    ax_rate.set_xlabel("Iteration")
    ax_rate.set_ylabel("negative-$\\rho$ fraction")
    ax_rate.grid(alpha=0.25)
    ax_rate.legend(loc="best")

    fig.tight_layout()

    filename = (
        "newton_model_agreement_ratio.png"
        if result["method"] == "Newton"
        else "newton_classic_model_agreement_ratio.png"
    )
    _save_or_show(fig, filename)



def plot_runtime_comparison(results):
    """Compare measured wall-clock time on the currently selected problem."""
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    methods = list(results.keys())
    times = [max(results[m].get("elapsed_time", 0.0), 1e-12) for m in methods]

    ax.bar(methods, times)
    ax.set_yscale("log")
    ax.set_xlabel("Method")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title(
        f"Runtime comparison — {EXPERIMENT_PROBLEM}, n={PROBLEM_DIM}"
    )
    ax.grid(axis="y", alpha=0.3)

    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.3g}s", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, "runtime_comparison.png")


def print_runtime_breakdown(results):
    print("\n" + "=" * 96)
    print(
        f"RUNTIME SUMMARY — problem={EXPERIMENT_PROBLEM}, "
        f"dimension={PROBLEM_DIM}"
    )
    print("=" * 96)
    print(
        f"{'method':>10s} | {'steps':>7s} | {'total(s)':>10s} | "
        f"{'line-search':>11s} | {'H-build':>9s} | "
        f"{'H-check':>9s} | {'linear solve':>12s}"
    )
    print("-" * 96)
    for method, result in results.items():
        line_search_text = (
            f"{result['line_search_time']:11.4f}"
            if "line_search_time" in result else f"{'—':>11s}"
        )
        print(
            f"{method:>10s} | "
            f"{result['num_steps']:7d} | "
            f"{result.get('elapsed_time', np.nan):10.4f} | "
            f"{line_search_text} | "
            f"{result.get('hessian_build_time', 0.0):9.4f} | "
            f"{result.get('hessian_check_time', 0.0):9.4f} | "
            f"{result.get('linear_solve_time', 0.0):12.4f}"
        )
    print("-" * 96)
    print(
        "Note: Newton's H-check column is the eigenvalue test used by the "
        "current modified-Newton safeguard; it is not part of plain Newton."
    )


# ============================================================
# 9. Run， only have one initial point
# ============================================================

if __name__ == "__main__":
    print(
        f"Running problem={EXPERIMENT_PROBLEM}, dimension={PROBLEM_DIM}, "
        f"methods={METHODS}"
    )

    results = {}
    failures = {}
    for method in METHODS:
        print(f"\nStarting method={method}")
        try:
            if method == "Newton":
                result = newton_method(
                    x0=X0,
                    max_iter=MAX_ITER,
                    grad_tol=GRAD_TOL,
                    absolute_grad_tol=ABSOLUTE_GRAD_TOL,
                )
            elif method == "Newton-classic":
                result = classical_newton_method(
                    x0=X0,
                    max_iter=MAX_ITER,
                    grad_tol=GRAD_TOL,
                    absolute_grad_tol=ABSOLUTE_GRAD_TOL,
                )
            else:
                result = nonlinear_cg(
                    x0=X0,
                    method=method,
                    max_iter=MAX_ITER,
                    grad_tol=GRAD_TOL,
                    absolute_grad_tol=ABSOLUTE_GRAD_TOL,
                    use_restart_552=USE_RESTART_552,
                    restart_nu=RESTART_NU,
                )
        except RuntimeError as exc:
            failures[method] = str(exc)
            print(f"[METHOD FAILED] {method}: {exc}")
            print("Continuing with the next method for debugging.")
            continue

        results[method] = result

        if SAVE_ITERATION_CSV:
            csv_path = save_iteration_csv(result)
            print(f"Saved {method} iterations to {csv_path}")

        if PRINT_ITERATION_TABLES and PROBLEM_DIM == 2:
            print_iteration_table(result)
        if PROBLEM_DIM == 2:
            plot_three_separate_views(result)

    if SAVE_SUMMARY_CSV:
        summary_path = save_summary_csv(results)
        print(f"Saved method summary to {summary_path}")

    if SHOW_CONVERGENCE_COMPARISON:
        plot_convergence_comparison(results)

    if SHOW_RESIDUAL_CONVERGENCE :
        plot_residual_convergence(results)

    if SHOW_DIRECTION_COSINES and results:
        plot_direction_cosines(results)

    if SHOW_NEWTON_MODEL_AGREEMENT:
        for method, result in results.items():
            if is_newton_method(method):
                plot_newton_model_agreement(result)

    print_runtime_breakdown(results)

    if failures:
        print("\nFAILED METHODS")
        for method, message in failures.items():
            print(f"  {method}: {message}")

    if SHOW_RUNTIME_COMPARISON:
        plot_runtime_comparison(results)

# ============================================================
# 10. Nonlinear CG experiment configuration and multi-start benchmark
# ============================================================

class NonlinearCGExperiment:
    """Configurable Rosenbrock solver and multi-start benchmark.
    """

    SUPPORTED_METHODS = (
        "SD", "Newton", "Newton-classic", "FR", "PR", "PR+", "HS",
        "FR-PR", "5.49", "5.50"
    )

    def __init__(
        self,
        dimension=100,
        methods=None,
        max_iter=1000,
        grad_tol=1e-5,
        absolute_grad_tol=1e-5,
        wolfe_c1=1e-4,
        wolfe_c2=0.1,
        line_search_maxiter=200,
        use_restart_552=True,
        restart_nu=0.1,
        newton_min_eigenvalue=1e-8,
        output_directory=None,
        save_results=True,
        save_plots=True,
        show_plots=False,
        performance_profile_max_factor=10.0,
    ):
        self.dimension = int(dimension)
        self.methods = list(methods or [
            "SD", "Newton", "FR", "PR+", "HS", "FR-PR", "5.49", "5.50"
        ])
        self.max_iter = int(max_iter)
        self.grad_tol = float(grad_tol)
        self.absolute_grad_tol = (
            None if absolute_grad_tol is None else float(absolute_grad_tol)
        )
        self.wolfe_c1 = float(wolfe_c1)
        self.wolfe_c2 = float(wolfe_c2)
        self.line_search_maxiter = int(line_search_maxiter)
        self.use_restart_552 = bool(use_restart_552)
        self.restart_nu = float(restart_nu)
        self.newton_min_eigenvalue = float(newton_min_eigenvalue)
        self.save_results = bool(save_results)
        self.save_plots = bool(save_plots)
        self.show_plots = bool(show_plots)
        self.performance_profile_max_factor = float(
            performance_profile_max_factor
        )
        if output_directory is None:
            output_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "nonlinear_cg_results",
                f"multistart_rosenbrock_n{self.dimension}",
            )
        self.output_directory = os.path.abspath(output_directory)
        self.figure_directory = os.path.join(self.output_directory, "figures")
        self.table_directory = os.path.join(self.output_directory, "tables")
        self._validate_configuration()

    def _validate_configuration(self):
        if self.dimension < 2:
            raise ValueError("dimension must be at least 2")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.grad_tol <= 0.0:
            raise ValueError("grad_tol must be positive")
        if self.absolute_grad_tol is not None and self.absolute_grad_tol <= 0.0:
            raise ValueError("absolute_grad_tol must be positive or None")
        if not (0.0 < self.wolfe_c1 < self.wolfe_c2 < 1.0):
            raise ValueError("Wolfe parameters must satisfy 0 < c1 < c2 < 1")
        if self.performance_profile_max_factor <= 1.0:
            raise ValueError("performance_profile_max_factor must exceed 1")
        unknown = [m for m in self.methods if m not in self.SUPPORTED_METHODS]
        if unknown:
            raise ValueError(f"unsupported methods: {unknown}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")

    def configuration(self):
        """Return a JSON-serializable snapshot of all experiment parameters."""
        return {
            "dimension": self.dimension,
            "methods": self.methods,
            "max_iter": self.max_iter,
            "grad_tol": self.grad_tol,
            "stopping_test": "||grad f||_inf < grad_tol * (1 + |f|)",
            "absolute_grad_tol_safeguard": self.absolute_grad_tol,
            "wolfe_c1": self.wolfe_c1,
            "wolfe_c2": self.wolfe_c2,
            "line_search_maxiter": self.line_search_maxiter,
            "use_restart_552": self.use_restart_552,
            "restart_nu": self.restart_nu,
            "newton_min_eigenvalue": self.newton_min_eigenvalue,
            "performance_profile_max_factor": (
                self.performance_profile_max_factor
            ),
        }

    @staticmethod
    def objective(x):
        return rosenbrock(x)

    @staticmethod
    def gradient(x):
        return rosenbrock_grad(x)

    @staticmethod
    def hessian(x):
        return rosenbrock_hessian(x)

    def initial_point(self, value=-1.2):
        return np.full(self.dimension, float(value), dtype=float)

    def generate_random_initial_points(self, count, low=-2.0, high=2.0, seed=0):
        count = int(count)
        if count < 1:
            raise ValueError("count must be positive")
        low_array = np.broadcast_to(np.asarray(low, dtype=float), (self.dimension,))
        high_array = np.broadcast_to(
            np.asarray(high, dtype=float), (self.dimension,)
        )
        if np.any(low_array >= high_array):
            raise ValueError("every lower bound must be smaller than its upper bound")
        rng = np.random.default_rng(seed)
        return rng.uniform(low_array, high_array, size=(count, self.dimension))



    def solve(self, method, x0):
        """Run the original procedural solver with this class configuration."""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"unsupported method: {method}")
        if method == "Newton":
            return newton_method(
                x0=x0,
                max_iter=self.max_iter,
                grad_tol=self.grad_tol,
                absolute_grad_tol=self.absolute_grad_tol,
                min_eigenvalue=self.newton_min_eigenvalue,
            )
        if method == "Newton-classic":
            return classical_newton_method(
                x0=x0,
                max_iter=self.max_iter,
                grad_tol=self.grad_tol,
                absolute_grad_tol=self.absolute_grad_tol,
            )
        return nonlinear_cg(
            x0=x0,
            method=method,
            max_iter=self.max_iter,
            grad_tol=self.grad_tol,
            use_restart_552=self.use_restart_552,
            restart_nu=self.restart_nu,
            absolute_grad_tol=self.absolute_grad_tol,
            wolfe_c1=self.wolfe_c1,
            wolfe_c2=self.wolfe_c2,
            line_search_maxiter=self.line_search_maxiter,
            print_failure_diagnostics=False,
        )

    def run_methods(self, x0):
        """Run every configured method and keep failures independent."""
        results, failures = {}, {}
        for method in self.methods:
            try:
                results[method] = self.solve(method, x0)
            except OptimizationFailure as exc:
                failures[method] = {
                    "status": exc.status,
                    "iteration": exc.iteration,
                    "message": str(exc),
                }
            except (
                FloatingPointError,
                OverflowError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as exc:
                failures[method] = {
                    "status": "runtime_failure",
                    "iteration": "",
                    "message": str(exc),
                }
        return results, failures

    def _save_multistart_rows(self, rows):
        os.makedirs(self.table_directory, exist_ok=True)
        path = os.path.join(self.table_directory, "multistart_results.csv")
        fieldnames = [
            "start_id", "method", "initial_f", "initial_norm",
            "status", "converged", "num_steps", "iteration_cost",
            "final_f", "final_stopping_measure", "elapsed_time_s",
            "failure_iteration", "failure_message",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _save_matrix(self, filename, matrix):
        """Save a start-by-method matrix with readable method headers."""
        os.makedirs(self.table_directory, exist_ok=True)
        path = os.path.join(self.table_directory, filename)
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["start_id", *self.methods])
            for start_id, values in enumerate(np.asarray(matrix)):
                writer.writerow([start_id, *values])
        return path

    def plot_iteration_performance_profile(self, iteration_matrix):
        """Plot a Dolan-More profile using iteration count as the cost."""
        costs = np.asarray(iteration_matrix, dtype=float)
        if costs.shape[1] != len(self.methods):
            raise ValueError("iteration_matrix has the wrong method count")
        best = np.min(costs, axis=1)
        ratios = np.full_like(costs, np.inf)
        valid_problem = np.isfinite(best)
        ratios[valid_problem] = (
            costs[valid_problem] / best[valid_problem, np.newaxis]
        )
        finite_ratios = ratios[np.isfinite(ratios)]
        visible_ratios = finite_ratios[
            finite_ratios <= self.performance_profile_max_factor
        ]
        tau = np.unique(np.concatenate((
            [1.0], visible_ratios, [self.performance_profile_max_factor]
        )))
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        total = costs.shape[0]
        for j, method in enumerate(self.methods):
            profile = np.array([
                np.count_nonzero(ratios[:, j] <= value) / total
                for value in tau
            ])
            solved = np.count_nonzero(np.isfinite(costs[:, j]))
            ax.step(
                tau,
                profile,
                where="post",
                linewidth=2.0,
                label=f"{method} ({solved}/{total} solved)",
            )
        ax.set_xscale("log")
        ax.set_xlim(1.0, self.performance_profile_max_factor)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Within this factor of the best iteration count (log scale)")
        ax.set_ylabel("Proportion of initial points")
        ax.set_title(
            f"Iteration-count performance profile — Rosenbrock, "
            f"n={self.dimension}"
        )
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = os.path.join(
            self.figure_directory, "iteration_performance_profile.png"
        )
        if self.save_plots:
            os.makedirs(self.figure_directory, exist_ok=True)
            fig.savefig(path, dpi=220, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        return path, ratios

    def run_multistart(self, count, low=-2.0, high=2.0, seed=0):
        initial_points = self.generate_random_initial_points(
            count=count, low=low, high=high, seed=seed
        )
        iteration_matrix = np.full(
            (len(initial_points), len(self.methods)), np.inf, dtype=float
        )
        rows = []
        for start_id, x0 in enumerate(initial_points):
            initial_f = self.objective(x0)
            initial_norm = float(np.linalg.norm(x0))
            for method_index, method in enumerate(self.methods):
                failure_iteration = ""
                failure_message = ""
                try:
                    result = self.solve(method, x0)
                    converged = bool(result["converged"])
                    status = "converged" if converged else "max_iter"
                    num_steps = int(result["num_steps"])
                    if converged:
                        iteration_matrix[start_id, method_index] = max(
                            num_steps, 1
                        )
                    final_f = float(result["final_f"])
                    final_measure = float(result["final_stopping_measure"])
                    elapsed_time = float(result["elapsed_time"])
                except OptimizationFailure as exc:
                    status = exc.status
                    converged = False
                    num_steps = exc.iteration
                    final_f = final_measure = elapsed_time = np.nan
                    failure_iteration = exc.iteration
                    failure_message = str(exc)
                except (
                    FloatingPointError,
                    OverflowError,
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ) as exc:
                    status = "runtime_failure"
                    converged = False
                    num_steps = ""
                    final_f = final_measure = elapsed_time = np.nan
                    failure_message = str(exc)
                rows.append({
                    "start_id": start_id,
                    "method": method,
                    "initial_f": initial_f,
                    "initial_norm": initial_norm,
                    "status": status,
                    "converged": converged,
                    "num_steps": num_steps,
                    "iteration_cost": (
                        iteration_matrix[start_id, method_index]
                        if converged else "inf"
                    ),
                    "final_f": final_f,
                    "final_stopping_measure": final_measure,
                    "elapsed_time_s": elapsed_time,
                    "failure_iteration": failure_iteration,
                    "failure_message": failure_message,
                })

        table_path = iteration_path = ratio_path = config_path = None
        if self.save_results:
            os.makedirs(self.output_directory, exist_ok=True)
            np.save(
                os.path.join(self.output_directory, "random_initial_points.npy"),
                initial_points,
            )
            table_path = self._save_multistart_rows(rows)
            iteration_path = self._save_matrix(
                "iteration_counts.csv", iteration_matrix
            )
            metadata = self.configuration()
            metadata.update({
                "random_start_count": int(count),
                "random_seed": seed,
                "random_lower_bound": np.asarray(low).tolist(),
                "random_upper_bound": np.asarray(high).tolist(),
            })
            config_path = os.path.join(
                self.output_directory, "experiment_config.json"
            )
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(metadata, config_file, indent=2)
        profile_path, ratios = self.plot_iteration_performance_profile(
            iteration_matrix
        )
        if self.save_results:
            ratio_path = self._save_matrix(
                "performance_ratios.csv", ratios
            )
        return {
            "initial_points": initial_points,
            "rows": rows,
            "iteration_matrix": iteration_matrix,
            "performance_ratios": ratios,
            "profile_path": profile_path,
            "results_table_path": table_path,
            "iteration_counts_path": iteration_path,
            "performance_ratios_path": ratio_path,
            "configuration_path": config_path,
        }
