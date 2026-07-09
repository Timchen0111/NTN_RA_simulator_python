import numpy as np
import cvxpy as cp


def solve_group_selection_policy(
    weights,
    ps_by_group,
    sat_num=None,
    imbalance_epsilon=0.0,
    initial_policy=None,
    maxiter=500,
    tol=1e-9,
):
    """
    Solve the group-based satellite selection subproblem.

    Objective:
        max sum_g w_g sum_k a_g,k p_s^g,k

    Constraints:
        sum_k a_g,k = 1, a_g,k >= 0
        sum_k (sum_g w_g a_g,k p_s^g,k - p_bar/K)^2 <= imbalance_epsilon

    Returns:
        dict[group] -> K-dimensional probability vector A_g.
    """
    groups = [tuple(group) for group in weights.keys()]
    if len(groups) == 0:
        return {}

    if sat_num is None:
        first_group = groups[0]
        sat_num = len(ps_by_group[first_group])
    if sat_num <= 0:
        raise ValueError("sat_num must be positive.")

    w = np.array([float(weights[group]) for group in groups], dtype=float)
    ps_matrix = np.vstack([
        np.asarray(ps_by_group[group], dtype=float)
        for group in groups
    ])
    if ps_matrix.shape != (len(groups), sat_num):
        raise ValueError(
            f"ps_by_group shape {ps_matrix.shape} does not match "
            f"({len(groups)}, {sat_num})."
        )
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative.")
    if np.sum(w) <= 0:
        raise ValueError("sum of group weights must be positive.")
    if not np.all(np.isfinite(ps_matrix)):
        raise ValueError("ps_by_group contains non-finite values.")

    # Normalize weights defensively; generated tables should already sum to 1.
    w = w / np.sum(w)
    group_count = len(groups)

    initial_matrix = None
    if initial_policy is not None:
        x0_matrix = np.vstack([
            np.asarray(initial_policy[tuple(group)], dtype=float)
            for group in groups
        ])
        if x0_matrix.shape != (group_count, sat_num):
            raise ValueError(
                f"initial_policy shape {x0_matrix.shape} does not match "
                f"({group_count}, {sat_num})."
            )
        row_sums = np.sum(x0_matrix, axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("each initial_policy row must have positive sum.")
        initial_matrix = x0_matrix / row_sums

    a_var = cp.Variable((group_count, sat_num), nonneg=True)
    # Effective received contribution per satellite:
    # effective_load[k] = sum_g w_g * a_g,k * p_s^g,k.
    effective_load = cp.sum(cp.multiply(w[:, None] * ps_matrix, a_var), axis=0)
    p_bar = cp.sum(effective_load)

    constraints = [
        cp.sum(a_var, axis=1) == 1.0,
    ]
    if imbalance_epsilon <= 0:
        constraints.append(effective_load == (p_bar / sat_num))
    else:
        constraints.append(
            cp.sum_squares(effective_load - (p_bar / sat_num))
            <= float(imbalance_epsilon)
        )

    objective = cp.Maximize(p_bar)
    problem = cp.Problem(objective, constraints)
    if initial_matrix is not None:
        a_var.value = initial_matrix

    solve_errors = []
    for solver in ("CLARABEL", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            if solver == "SCS":
                problem.solve(
                    solver=solver,
                    warm_start=True,
                    max_iters=maxiter,
                    eps=tol,
                    verbose=False,
                )
            else:
                problem.solve(
                    solver=solver,
                    warm_start=True,
                    verbose=False,
                )
        except Exception as exc:
            solve_errors.append(f"{solver}: {exc}")
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
        solve_errors.append(f"{solver}: status={problem.status}")
    else:
        detail = "; ".join(solve_errors) if solve_errors else "no compatible solver installed"
        raise RuntimeError(f"Group selection optimization failed: {detail}")

    if a_var.value is None:
        raise RuntimeError(
            f"Group selection optimization failed: solver returned no value "
            f"(status={problem.status})."
        )

    a = np.clip(np.asarray(a_var.value, dtype=float), 0.0, 1.0)
    row_sums = np.sum(a, axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Group selection optimization returned an invalid policy.")
    a = a / row_sums
    effective = np.sum(w[:, None] * a * ps_matrix, axis=0)
    p_bar_value = float(np.sum(effective))
    imbalance = float(np.sum((effective - (p_bar_value / sat_num)) ** 2))
    if imbalance_epsilon <= 0:
        if not np.allclose(effective, np.ones(sat_num) * (p_bar_value / sat_num), atol=1e-5):
            raise RuntimeError(
                f"Group selection optimization violated effective-load balance constraint: "
                f"max error={np.max(np.abs(effective - (p_bar_value / sat_num)))}"
            )
    elif imbalance > float(imbalance_epsilon) + 1e-5:
        raise RuntimeError(
            f"Group selection optimization violated imbalance constraint: "
            f"{imbalance} > {imbalance_epsilon}"
        )

    return {
        group: a[idx].copy()
        for idx, group in enumerate(groups)
    }
