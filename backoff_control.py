import numpy as np
from scipy.optimize import minimize


class SatelliteEnv:
    def __init__(self, N_tilde, current_rho):
        self.n_tilde = N_tilde
        self.rho = current_rho

    def compute_C(self, p_b, p_c, D, p_s):
        C = np.zeros((D + 1, D + 1))
        for k in range(1, D + 1):
            for n in range(1, k + 1):
                prod = 1.0
                for j in range(n + 1, k + 1):
                    prod *= (1 - (1 - p_b[j - 1]) * p_s * (1 - p_c))
                C[k, n] = prod
        return C

    def compute_pi(self, C, D, p_d):
        numerator = np.zeros(D)
        inner_sum = 0.0
        for n in range(1, D + 1):
            val = sum(p_d[k - 1] * C[k, n] for k in range(n, D + 1))
            numerator[n - 1] = self.rho * val
            inner_sum += val

        # Pi is observed after packet arrivals and before the ACB decision.
        return numerator / ((1 - self.rho) + self.rho * inner_sum)

    def solve_p_c(self, p_b, D, p_d, p_s, K, Z):
        p_c = 0.5
        for _ in range(50):
            C = self.compute_C(p_b, p_c, D, p_s)
            pi = self.compute_pi(C, D, p_d)
            Lambda = self.n_tilde * np.sum(pi * (1 - p_b)) * p_s
            new_p_c = 1 - np.exp(-Lambda / (K * Z))
            if abs(new_p_c - p_c) < 1e-7:
                break
            p_c = new_p_c
        return p_c, C, pi


def priority_acb_backoff(lambda_by_state, total_preambles):
    lambda_by_state = np.asarray(lambda_by_state, dtype=float)
    D = len(lambda_by_state)
    if D == 0:
        return np.array([], dtype=float)

    if total_preambles <= 0:
        return np.ones(D, dtype=float)

    lambda_by_state = np.maximum(lambda_by_state, 0.0)
    if np.sum(lambda_by_state) <= 0 or not np.all(np.isfinite(lambda_by_state)):
        return np.zeros(D, dtype=float)

    state_index = np.arange(1, D + 1, dtype=float)
    urgency_weight = 1.0 / state_index
    weighted_load = urgency_weight * lambda_by_state
    weighted_total = np.sum(weighted_load)
    if weighted_total <= 0 or not np.isfinite(weighted_total):
        return np.zeros(D, dtype=float)

    target_ratio_by_state = weighted_load / weighted_total

    # Map delay state n to priority class ell(n) = D - n + 1.
    lambda_by_priority = lambda_by_state[::-1]
    target_ratio_by_priority = target_ratio_by_state[::-1]

    access_by_priority = np.ones(D, dtype=float)
    previous_adjusted_ratio = 0.0
    for priority_idx in range(D):
        adjusted_ratio = target_ratio_by_priority[priority_idx]
        if priority_idx > 0 and access_by_priority[priority_idx - 1] >= 1.0 - 1e-12:
            unused_ratio = previous_adjusted_ratio - (
                lambda_by_priority[priority_idx - 1] / total_preambles
            )
            adjusted_ratio += max(0.0, unused_ratio)

        class_load = lambda_by_priority[priority_idx]
        if class_load <= 0:
            access_by_priority[priority_idx] = 1.0
        else:
            access_by_priority[priority_idx] = min(
                1.0,
                total_preambles * adjusted_ratio / class_load,
            )
        previous_adjusted_ratio = adjusted_ratio

    access_by_state = access_by_priority[::-1]
    return np.clip(1.0 - access_by_state, 0.0, 1.0)


def backoff_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z, backoff_mode, Lambda):
    if backoff_mode == 1:
        return proposed_backoff_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z)

    if backoff_mode == 2:
        return dynamic_acb_backoff(N_tilde, rho, D, p_d, p_s, K, Z)

    if backoff_mode == 3:
        return priority_acb_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z)

    raise ValueError(f"Unsupported backoff mode: {backoff_mode}")


def proposed_backoff_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z):
    env = SatelliteEnv(N_tilde, rho)

    def objective(p_b_vec):
        p_c, _, _ = env.solve_p_c(p_b_vec, D, p_d, p_s, K, Z)
        return get_loss(p_b_vec, p_c, p_s, p_d, D)

    res = minimize(
        objective,
        last_p_b,
        method="L-BFGS-B",
        bounds=[(0, 1)] * D,
        tol=1e-6,
    )
    opt_p_b = res.x
    _, _, opt_pi = env.solve_p_c(opt_p_b, D, p_d, p_s, K, Z)
    return opt_p_b, opt_pi


def dynamic_acb_backoff(N_tilde, rho, D, p_d, p_s, K, Z):
    env = SatelliteEnv(N_tilde, rho)
    total_preambles = K * Z
    if N_tilde <= 0:
        access_prob = 1.0
    else:
        access_prob = min(1.0, total_preambles / N_tilde)

    backoff_prob = 1.0 - access_prob
    backoff = backoff_prob * np.ones(D)
    _, _, opt_pi = env.solve_p_c(backoff, D, p_d, p_s, K, Z)
    return backoff, opt_pi


def priority_acb_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z):
    env = SatelliteEnv(N_tilde, rho)
    _, _, pi_from_previous_policy = env.solve_p_c(last_p_b, D, p_d, p_s, K, Z)
    lambda_by_state = N_tilde * pi_from_previous_policy
    backoff = priority_acb_backoff(lambda_by_state, K * Z)
    _, _, opt_pi = env.solve_p_c(backoff, D, p_d, p_s, K, Z)
    return backoff, opt_pi


def get_loss(p_b, p_c, p_s, p_d_arr, D):
    L = 0.0
    for i in range(1, D + 1):
        prod = 1.0
        for j in range(1, i + 1):
            prod *= (1 - (1 - p_b[j - 1]) * p_s * (1 - p_c))
        L += p_d_arr[i - 1] * prod
    return L
