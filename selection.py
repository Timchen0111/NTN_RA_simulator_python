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
    def __init__(self, satellite_list, S_max=10, mem_length=5):
        self.sat_num = len(satellite_list)
        self.S_max = S_max
        self.mem_length = mem_length
        
        # 修正維度定義：確保這裡是固定的
        self.state_dim = mem_length * (self.sat_num * 2) 
        self.action_dim = self.sat_num * 3 
        
        # 修正 2: 引入目標網路 (Target Network)
        self.policy_net = SimpleDQN(self.state_dim, self.action_dim)
        self.target_net = SimpleDQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # 修正 1: 經驗回放池 (Replay Buffer)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32 #這個可以調大
        
        self.load_history = deque(maxlen=mem_length)
        self.score_history = deque(maxlen=mem_length)
        self.current_S = np.ones(self.sat_num)
        self.steps_done = 0

    def get_state(self, current_load):
        # 使用 deque 自動維護長度
        self.load_history.append(current_load)
        self.score_history.append(self.current_S.copy())
        
        # 冷啟動處理：如果紀錄不夠長，前面補零
        loads = list(self.load_history)
        scores = list(self.score_history)
        
        while len(loads) < self.mem_length:
            loads.insert(0, np.zeros(self.sat_num))
            scores.insert(0, np.zeros(self.sat_num))
            
        state = np.hstack([np.array(loads).flatten(), 
                           np.array(scores).flatten()])
        
        # 強制確認維度正確，否則報錯時會很清楚
        assert state.shape[0] == self.state_dim, f"Dimension mismatch! Expected {self.state_dim}, got {state.shape[0]}"
        return state

    def select_action(self, state, epsilon):  ##這是主要供main調用的程式  #使用 epsilon-greedy 策略選擇動作，epsilon需要隨時間衰減以平衡探索與利用
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
        self._apply_action(action_idx)
        return action_idx, self.current_S # 多回傳一個 idx 方便存入 memory

    def _apply_action(self, action_idx):
        target_sat_idx = action_idx // 3
        adjustment = (action_idx % 3) - 1
        self.current_S[target_sat_idx] += adjustment
        self.current_S = np.clip(self.current_S, 1, self.S_max)

    def compute_reward(self, current_load):
        """
        對應論文 Reward: 負載方差的負值
        """
        avg_load = np.mean(current_load)
        variance = np.mean((current_load - avg_load)**2)
        return -variance

    def store_transition(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 修正：從 Replay Buffer 隨機抽樣
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next = zip(*batch)
        
        s_t = torch.FloatTensor(np.array(s))
        a_t = torch.LongTensor(np.array(a)).view(-1, 1)
        r_t = torch.FloatTensor(np.array(r)).view(-1, 1)
        s_next_t = torch.FloatTensor(np.array(s_next))
        
        # 使用 policy_net 計算目前的 Q
        current_q = self.policy_net(s_t).gather(1, a_t)
        
        # 使用 target_net 計算目標 Q
        with torch.no_grad():
            max_next_q = self.target_net(s_next_t).max(1)[0].view(-1, 1)
            target_q = r_t + 0.99 * max_next_q
            
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期同步 Target Network
        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())