import numpy as np

class satellite:
    def __init__(self, id, location, t_max=600, Z=54):
        self.id = id
        self.location = location
        self.t_max = t_max  # Maximum pass duration [cite: 86]
        self.Z = Z          # Number of available preambles [cite: 34]
        self.ue_buffer = [] # 暫存這個 Time Slot 嘗試接入的 UE

    def get_remaining_time(self, current_time):
        # 模擬剩餘覆蓋時間 gamma [cite: 86]
        # 簡單假設：時間越久，剩餘時間越少
        passed_time = current_time % self.t_max
        return self.t_max - passed_time

    def receive_preamble(self, ue_id):
        """接收 UE 發送的 Preamble"""
        # 模擬 UE隨機選取一個 Preamble (0 到 Z-1)
        chosen_preamble = np.random.randint(0, self.Z)
        self.ue_buffer.append((ue_id, chosen_preamble))

    def check_RA_success(self):
        """
        模擬碰撞檢測 [cite: 39, 41]
        如果有兩個以上的 UE 選到同一個 Preamble，則發生碰撞。
        回傳：成功接入的 UE ID 列表
        """
        if not self.ue_buffer:
            return []

        # 統計每個 Preamble 被選取的次數
        preamble_counts = {}
        ue_map = {} # 記錄誰選了哪個 Preamble

        for ue_id, preamble in self.ue_buffer:
            if preamble not in preamble_counts:
                preamble_counts[preamble] = 0
                ue_map[preamble] = []
            preamble_counts[preamble] += 1
            ue_map[preamble].append(ue_id)

        successful_ues = []
        # 只有當計數為 1 時才算成功 (Singleton slot)
        for preamble, count in preamble_counts.items():
            if count == 1:
                successful_ues.append(ue_map[preamble][0])
        
        # 清空 Buffer 準備下一個 Slot
        self.ue_buffer = []
        return successful_ues

    def get_info(self):
        print(f"ID: {self.id}, Location: {self.location}, Preambles: {self.Z}")

class UE:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.budget = 0       # d_B,i [cite: 36]
        self.delay = 0        # d_i [cite: 36]
        self.active = False   # 是否有封包要傳
        self.visible_satellites = [] # K_i [cite: 87]
        self.target_satellite = None # 決定要嘗試哪顆衛星
        self.acb_factor = 0.0 # p_i,k [cite: 46]

    def acquire_visible_sat(self, satellite_list):
        """簡單模擬：將所有衛星視為可見 (可加入距離判斷)"""
        self.visible_satellites = satellite_list
        # Report[cite: 87]: K_i is number of visible satellites

    def new_packet(self, budget):
        """產生新封包 [cite: 6]"""
        self.active = True
        self.budget = budget
        self.delay = 0
        self.target_satellite = None

    def calculate_ACB(self, current_time, p=4, x_params=[1, 2, 0.05]):
        """
        核心邏輯：計算混合急迫性分數 S 與 ACB 因子
        Eq (6): S_{i,k} = (|d/d_B|^p + |1/K * (1-gamma/T_max)|^p)^(1/p)
        Eq (5): p_ACB = x1 * S^x2 + x3
        """
        if not self.active or not self.visible_satellites:
            return

        best_score = -1
        best_sat = None
        
        K_i = len(self.visible_satellites)
        x1, x2, x3 = x_params

        # 針對每顆衛星計算分數，並選擇最高分的衛星進行嘗試 (或是依策略排序)
        # Report[cite: 6]: "(3) Determine the satellite attempt order"
        
        for sat in self.visible_satellites:
            gamma = sat.get_remaining_time(current_time) # remaining coverage time [cite: 86]
            t_max = sat.t_max

            # Term 1: Delay Urgency
            term1 = (self.delay / self.budget) ** p
            
            # Term 2: Coverage Urgency
            term2 = ( (1/K_i) * (1 - gamma/t_max) ) ** p # [cite: 83]

            # Calculate S (Score)
            S = (term1 + term2) ** (1/p)

            # Calculate p_ACB (Probability) [cite: 64]
            p_acb = x1 * (S ** x2) + x3
            
            # 限制在 [0, 1]
            p_acb = max(0.0, min(1.0, p_acb))

            # 策略：這裡簡單假設 UE 選擇算出來機率最高的衛星
            if S > best_score:
                best_score = S
                best_sat = sat
                self.acb_factor = p_acb

        self.target_satellite = best_sat

    def ACB_test(self):
        """
        執行 ACB 測試 (Bernoulli Trial) [cite: 44]
        X_{i,k} ~ Bernoulli(p_{i,k})
        """
        if not self.active or self.target_satellite is None:
            return False
        
        rand_val = np.random.rand()
        if rand_val <= self.acb_factor:
            return True # Passed
        else:
            return False # Barred

    def execute_RA(self):
        """如果通過測試，則發送 Preamble [cite: 6, 39]"""
        if self.target_satellite:
            self.target_satellite.receive_preamble(self.id)

    def update_status(self, success_list):
        """更新狀態：成功則重置，失敗則延遲 +1"""
        if not self.active:
            return

        if self.id in success_list:
            # 成功接入 [cite: 36]
            self.active = False
            self.delay = 0
            # print(f"UE {self.id} Success!")
        else:
            # 失敗或被擋下，延遲增加
            self.delay += 1
            # Check Packet Drop 
            if self.delay > self.budget:
                self.active = False # Packet dropped
                # print(f"UE {self.id} Packet Dropped!")

    def get_info(self):
        status = "Active" if self.active else "Idle"
        print(f"UE {self.id} [{status}] | Loc: {self.location} | Budget: {self.budget} | Delay: {self.delay} | p_ACB: {self.acb_factor:.2f}")

# --- 簡單模擬測試 (Actionable Simulation) ---

# 1. 初始化
sat1 = satellite(id=1, location=[0,0,500])
ue1 = UE(id=101, location=[10,20,0])

# 2. 設定環境
ue1.acquire_visible_sat([sat1])
ue1.new_packet(budget=10) # 設定 Delay Budget

# 3. 模擬一個 Time Slot
current_time = 100

# Step A: 計算參數
ue1.calculate_ACB(current_time)
ue1.get_info()

# Step B: 執行 ACB 測試
if ue1.ACB_test():
    print("ACB Check Passed! Transmitting...")
    ue1.execute_RA()
else:
    print("ACB Check Failed. Waiting...")

# Step C: 衛星端檢查結果
success_ids = sat1.check_RA_success()

# Step D: 更新 UE 狀態
ue1.update_status(success_ids)