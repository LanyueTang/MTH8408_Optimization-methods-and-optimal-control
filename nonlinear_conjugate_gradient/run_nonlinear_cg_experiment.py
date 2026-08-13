"""Editable main program for single-problem, multi-start comparisons.
"""

from collections import Counter
from pathlib import Path

from nonlinear_cg_rosenbrock_large import NonlinearCGExperiment


# ============================================================================
# CONFIGURATION 
# ============================================================================

# Problem and methods
PROBLEM_DIMENSION = 100
METHODS = ["SD", "Newton-classic", "FR", "PR+", "HS", "FR-PR"]


MAX_ITER = 1000
GRAD_TOL = 1e-5
ABSOLUTE_GRAD_TOL = 1e-5

# Strong-Wolfe line search used by SD and nonlinear-CG methods
WOLFE_C1 = 1e-4
WOLFE_C2 = 0.1
LINE_SEARCH_MAXITER = 200

# Nonlinear-CG restart, Eq. (5.52)
USE_RESTART_552 = True
RESTART_NU = 0.1

NEWTON_MIN_EIGENVALUE = 1e-8

# Random multi-start experiment
NUM_RANDOM_STARTS = 200
RANDOM_SEED = 2026
RANDOM_LOWER_BOUND = -2.0
RANDOM_UPPER_BOUND = 2.0


PERFORMANCE_PROFILE_MAX_FACTOR = 5

SAVE_RESULTS = True
SAVE_PLOTS = True
SHOW_PLOTS = False

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "nonlinear_cg_results"
    / (
        f"multistart_rosenbrock_n{PROBLEM_DIMENSION}"
        f"_m{NUM_RANDOM_STARTS}_seed{RANDOM_SEED}"
    )
)


def main():
    experiment = NonlinearCGExperiment(
        dimension=PROBLEM_DIMENSION,
        methods=METHODS,
        max_iter=MAX_ITER,
        grad_tol=GRAD_TOL,
        absolute_grad_tol=ABSOLUTE_GRAD_TOL,
        wolfe_c1=WOLFE_C1,
        wolfe_c2=WOLFE_C2,
        line_search_maxiter=LINE_SEARCH_MAXITER,
        use_restart_552=USE_RESTART_552,
        restart_nu=RESTART_NU,
        newton_min_eigenvalue=NEWTON_MIN_EIGENVALUE,
        output_directory=OUTPUT_DIRECTORY,
        save_results=SAVE_RESULTS,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        performance_profile_max_factor=PERFORMANCE_PROFILE_MAX_FACTOR,
    )

    report = experiment.run_multistart(
        count=NUM_RANDOM_STARTS,
        low=RANDOM_LOWER_BOUND,
        high=RANDOM_UPPER_BOUND,
        seed=RANDOM_SEED,
    )

    print("\nMULTI-START STATUS SUMMARY")
    print(
        f"dimension={PROBLEM_DIMENSION}, starts={NUM_RANDOM_STARTS}, "
        f"seed={RANDOM_SEED}"
    )
    for method in METHODS:
        statuses = Counter(
            row["status"] for row in report["rows"]
            if row["method"] == method
        )
        status_text = ", ".join(
            f"{status}={count}" for status, count in sorted(statuses.items())
        )
        print(f"  {method:>8s}: {status_text}")

    print(f"\nResults directory: {OUTPUT_DIRECTORY}")
    print(f"Performance profile: {report['profile_path']}")
    print(f"Detailed results: {report['results_table_path']}")
    print(
        "Failures and max-iteration runs have infinite performance ratios; "
        "they remain in the denominator of the profile."
    )


if __name__ == "__main__":
    main()
