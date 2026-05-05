import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- RL_network.py 內容 ---
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

# --- Agent 整合類別 ---
class SatelliteSelectionAgent:
    def __init__(self, satellite_list, S_max=10, mem_length=5):
        self.sat_num = len(satellite_list)
        self.S_max = S_max
        self.mem_length = mem_length
        
        # 1. 定義維度 (對應論文中的 State 與 Action)
        # State: 歷史負載 + 歷史分數 (每個時隙包含 2*sat_num 個數值)
        self.state_dim = mem_length * (self.sat_num * 2) 
        # Action: 離散化動作，例如每顆衛星可選擇 {分數-1, 不變, 分數+1}
        self.action_dim = self.sat_num * 3 
        
        # 2. 初始化 DQN 網路與優化器
        self.policy_net = SimpleDQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # 3. 狀態空間與目前分數
        self.load_history = [] 
        self.score_history = []
        self.current_S = np.ones(self.sat_num) # 初始得分向量 S

    def get_state(self, current_load):
        """
        維護 FIFO 隊列並產生一維狀態向量
        """
        self.load_history.append(current_load)
        self.score_history.append(self.current_S.copy())
        
        if len(self.load_history) > self.mem_length:
            self.load_history.pop(0)
            self.score_history.pop(0)
            
        # 組合負載與分數歷史，對應論文 \mathcal{H}^{m-1}
        state = np.hstack([np.array(self.load_history).flatten(), 
                           np.array(self.score_history).flatten()])
        return state

    def compute_reward(self, current_load):
        """
        對應論文 Reward: 負載方差的負值
        """
        avg_load = np.mean(current_load)
        variance = np.mean((current_load - avg_load)**2)
        return -variance

    def select_action(self, state, epsilon=0.1):
        """
        實現策略 Phi: Input State -> Output Score S
        """
        # Epsilon-greedy 策略用於exploration
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
        # 將離散索引轉為分數更新
        self._apply_action(action_idx)
        return self.current_S

    def _apply_action(self, action_idx):
        """
        將神經網路輸出的 index 映射到具體的衛星分數調整
        """
        target_sat_idx = action_idx // 3
        adjustment = (action_idx % 3) - 1 # 映射到 {-1, 0, 1}
        
        self.current_S[target_sat_idx] += adjustment
        # 限制分數在 [1, S_max] 區間
        self.current_S = np.clip(self.current_S, 1, self.S_max)

    def update_policy(self, state, action_idx, reward, next_state):
        """
        基本的 DQN 更新邏輯
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        reward_t = torch.FloatTensor([reward])
        
        # 計算目前的 Q 值
        current_q = self.policy_net(state_t)[0][action_idx]
        
        # 計算目標 Q 值 (簡化版：Reward + gamma * max_next_Q)
        with torch.no_grad():
            max_next_q = torch.max(self.policy_net(next_state_t))
            target_q = reward_t + 0.99 * max_next_q
            
        # 更新權重 theta
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()