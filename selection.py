import numpy as np
import random
from collections import deque
import cvxpy as cp

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SatelliteSelectionAgent requires PyTorch. The current convex "
            "selection path does not use it, so torch is not in requirements.txt."
        ) from exc
    return torch, nn, F, optim


def _build_simple_dqn(state_dim, action_dim):
    _, nn, F, _ = _require_torch()

    class SimpleDQN(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SimpleDQN, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return SimpleDQN(state_dim, action_dim)

class SatelliteSelectionAgent:
    def __init__(self, satellite_list, S_max, mem_length=5):
        self.sat_num = len(satellite_list)
        self.S_max = S_max
        self.mem_length = mem_length
        
        # State: 負載 (sat_num) + 幾何特徵 TTG (sat_num)
        # 移除舊分數歷史，將維度集中在物理特徵上
        self.state_dim = mem_length * (self.sat_num * 2) 
        
        # Action: 調整哪顆星 (sat_num) 以及調整方向 (-1, 0, +1)
        self.action_dim = self.sat_num * 3 
        
        self.torch, self.nn, self.F, self.optim = _require_torch()
        self.policy_net = _build_simple_dqn(self.state_dim, self.action_dim)
        self.target_net = _build_simple_dqn(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = self.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=5000) # 稍微加大 Buffer 應對高噪聲
        self.batch_size = 64 # 加大 Batch 增加訓練穩定性
        
        self.load_history = deque(maxlen=mem_length)
        self.ttg_history = deque(maxlen=mem_length) # 新增幾何特徵歷史
        
        self.current_S = np.ones(self.sat_num)
        self.steps_done = 0

    def get_state(self, current_load, current_elevation_angle):
        """
        current_elevation_angle: 各衛星相對於熱點中心(如台北)的仰角特徵
        """
        self.load_history.append(current_load)
        self.ttg_history.append(current_elevation_angle)
        
        loads = list(self.load_history)
        ttgs = list(self.ttg_history)
        
        # 冷啟動填充
        while len(loads) < self.mem_length:
            loads.insert(0, np.zeros(self.sat_num))
            ttgs.insert(0, np.zeros(self.sat_num))
            
        # 拼接特徵：[Load_t, TTG_t, Load_t-1, TTG_t-1, ...]
        state = np.hstack([np.array(loads).flatten(), 
                           np.array(ttgs).flatten()])
        
        return state

    def select_action(self, state, epsilon, S_heuristic=None):
        """
        S_heuristic: 可選參數。如果傳入 MODE 4 的分數，RL 將在此基礎上進行微調
        """
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = self.torch.FloatTensor(state).unsqueeze(0)
            with self.torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = self.torch.argmax(q_values).item()
        
        # TO DO：套用啟發式分數作為初始 S，讓 RL 主要學習微調而非從零開始
        if S_heuristic is not None:
            self.current_S = S_heuristic.copy()

        self._apply_action(action_idx)
        return action_idx, self.current_S

    def _apply_action(self, action_idx):
        target_sat_idx = action_idx // 3
        adjustment = (action_idx % 3) - 1 # {-1, 0, +1}
        
        self.current_S[target_sat_idx] += adjustment
        self.current_S = np.clip(self.current_S, 1, self.S_max)

    def compute_reward(self, current_load):
        avg_load = np.mean(current_load)
        variance = np.mean((current_load - avg_load)**2)
        return -variance

    def store_transition(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next = zip(*batch)
        
        s_t = self.torch.FloatTensor(np.array(s))
        a_t = self.torch.LongTensor(np.array(a)).view(-1, 1)
        r_t = self.torch.FloatTensor(np.array(r)).view(-1, 1)
        s_next_t = self.torch.FloatTensor(np.array(s_next))
        
        # Double DQN 邏輯
        current_q = self.policy_net(s_t).gather(1, a_t)
        
        with self.torch.no_grad():
            # 由 Policy Net 選出最佳動作的 Index
            next_actions = self.policy_net(s_next_t).argmax(dim=1, keepdim=True)
            # 由 Target Net 評估該動作的價值，減少高估偏誤
            max_next_q = self.target_net(s_next_t).gather(1, next_actions)
            target_q = r_t + 0.99 * max_next_q
            
        loss = self.F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪抗噪
        self.torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    def reset_history(self):
        """在每個 Episode 開始前清空歷史緩存"""
        self.load_history.clear()
        self.ttg_history.clear()
        self.current_S = np.ones(self.sat_num) # 重置分數為 1
        self.steps_done = 0


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
