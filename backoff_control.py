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
                    prod *= (1 - (1 - p_b[j-1]) * p_s * (1 - p_c))
                C[k, n] = prod
        return C

    def compute_pi(self, C, D, p_d):
        
        numerator = np.zeros(D)
        inner_sum = 0.0
        for n in range(1, D + 1):
            val = sum(p_d[k-1] * C[k, n] for k in range(n, D + 1))
            numerator[n-1] = self.rho * val
            inner_sum += val
        return numerator / (1 + self.rho * inner_sum)
        

    def solve_p_c(self, p_b, D, p_d, p_s, K, Z):
        p_c = 0.5
        for _ in range(50):
            C = self.compute_C(p_b, p_c, D, p_s)
            pi = self.compute_pi(C, D, p_d)
            Lambda = self.n_tilde * np.sum(pi * (1 - p_b))
            new_p_c = 1 - np.exp(-Lambda / (K * Z))
            if abs(new_p_c - p_c) < 1e-7: break
            p_c = new_p_c
        return p_c, C, pi

def backoff_control(N_tilde, last_p_b, rho, D, p_d, p_s, K, Z, MODE, Lambda):
    if MODE != 3 and MODE != 7:
        env = SatelliteEnv(N_tilde, rho)
        def objective(p_b_vec):
            p_c, _, _ = env.solve_p_c(p_b_vec, D, p_d, p_s, K, Z)
            return get_loss(p_b_vec, p_c, p_d, D)

        res = minimize(objective, last_p_b, method='L-BFGS-B', 
                    bounds=[(0, 1)] * D, tol=1e-6)
        opt_p_b = res.x
        _, _, opt_pi = env.solve_p_c(opt_p_b, D, p_d, p_s, K, Z)
        return opt_p_b, opt_pi
    else:
        # MODE 3 or MODE 7: 使用非狀態依賴的固定 backoff probability
        env = SatelliteEnv(N_tilde, rho)
        backoff_prob = Lambda / (K * Z)  # Non-state dependent backoff probability
        backoff =  backoff_prob * np.ones(D)  # 所有 delay budget 使用相同的 backoff probability
        p_c, _, opt_pi = env.solve_p_c(backoff, D, p_d, p_s, K, Z)
        return backoff, opt_pi

def get_loss(p_b, p_c, p_d_arr, D):
    L = 0.0
    for i in range(1, D + 1):
        prod = 1.0
        for j in range(1, i + 1):
            prod *= (1 - (1 - p_b[j-1]) * (1 - p_c))
        L += p_d_arr[i-1] * prod
    return L