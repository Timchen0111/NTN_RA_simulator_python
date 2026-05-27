import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

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
        
        self.policy_net = SimpleDQN(self.state_dim, self.action_dim)
        self.target_net = SimpleDQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
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
        
        s_t = torch.FloatTensor(np.array(s))
        a_t = torch.LongTensor(np.array(a)).view(-1, 1)
        r_t = torch.FloatTensor(np.array(r)).view(-1, 1)
        s_next_t = torch.FloatTensor(np.array(s_next))
        
        # Double DQN 邏輯
        current_q = self.policy_net(s_t).gather(1, a_t)
        
        with torch.no_grad():
            # 由 Policy Net 選出最佳動作的 Index
            next_actions = self.policy_net(s_next_t).argmax(dim=1, keepdim=True)
            # 由 Target Net 評估該動作的價值，減少高估偏誤
            max_next_q = self.target_net(s_next_t).gather(1, next_actions)
            target_q = r_t + 0.99 * max_next_q
            
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪抗噪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
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