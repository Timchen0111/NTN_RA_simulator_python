import numpy as np
class satellite:
    def __init__(self, id, location, t_max=600, Z=54):
        self.id = id
        self.location = location
        self.t_max = t_max  # Maximum pass duration 在套用真實衛星軌道前先隨便定
        self.Z = Z          # Number of available preambles 固定為54
        self.ue_pre = {} # 暫存這個 Time Slot 嘗試接入的 UE
    def receive_preamble(self,ue_id):
        # 模擬 UE隨機選取一個 Preamble (0 到 Z-1)
        chosen_preamble = np.random.randint(0, self.Z)
        self.ue_pre[ue_id] = chosen_preamble
    def check_RA_success(self):
        seen_values = set()
        duplicates = set()
        success_list = []
        for value in self.ue_pre.values():
            if value in seen_values:
                duplicates.add(value)
            else:
                seen_values.add(value)
        for ue in self.ue_pre.keys():
            if self.ue_pre[ue] not in duplicates:
                success_list.append(ue)
        self.ue_pre.clear()
        return success_list

    def get_info(self):
        print("ID:", self.id, "Location:", self.location)

class UE:
    def __init__(self,location,id):
        self.id = id
        self.location = location
        self.budget = 0
        self.delay = 0
        self.loss = 0
        self.success = 0
        self.active = False #Boolean
        self.active_prob = 0.5
        self.visible_satellites = []
        self.order = []
        self.ACB = []
    def acquire_visible_sat(self,sat_list):
        self.visible_satellites = sat_list #先簡單設成全部
    def new_time(self):
        if self.active == True:
            self.delay += 1
            if self.delay > self.budget:
                self.active = False
                self.loss += 1
                self.delay = 0
        else:
            if np.random.rand()<self.active_prob: 
                self.new_packet()
    def new_packet(self):
        self.active = True
        self.budget = 10 #之後會改
        self.delay = 0
        self.target_satellite = None
    def calculate_ACB(self, current_time, p=4, x_params=[1, 2, 0.05]):
        """
        核心邏輯：計算混合急迫性分數 S 與 ACB 因子
        Eq (6): S_{i,k} = (|d/d_B|^p + |1/K * (1-gamma/T_max)|^p)^(1/p)
        Eq (5): p_ACB = x1 * S^x2 + x3
        if not self.active or not self.visible_satellites:
            return

        K_i = len(self.visible_satellites)
        x1, x2, x3 = x_params
        ACB_set = []
        for sat in self.visible_satellites:
            gamma = sat.get_remaining_time(current_time) # 這裡假設所有UE可視時間相同，之後必須換掉
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
            ACB_set.append(p_acb)
        """
        ACB_set = []
        for sat in self.visible_satellites:
            ACB_set.append(0.5) #暫時先確定能動，先完成後續部分。`
        self.ACB = ACB_set
    def determine_order(self):
        self.order = np.random.permutation(len(self.visible_satellites)) #純隨機，之後再改
    def ACB_test(self):
        if not self.active:
            return
        chosen_sat = None
        pass_acb = False
        for i in self.order:
            p_acb = self.ACB[i]
            if np.random.rand()<p_acb:
                pass_acb = True
                chosen_sat = i
                break
        if pass_acb:
            # 取得對應的衛星物件
            target_sat = self.visible_satellites[chosen_sat]
            self.execute_RA(target_sat)
        else:
            # Backoff: 本回合不傳輸
            pass
    def execute_RA(self,target_sat):
        #實際傳輸 Preamble
        target_sat.receive_preamble(self.id)        
    def receive_feedback(self, success_list):
        #接收衛星回傳的結果並更新狀態
        if self.id in success_list:
            # 成功接入，重置狀態
            self.active = False
            self.success += 1
        else:
            # 碰撞失敗，保持 active，下回合 delay 會增加
            pass

    def get_info(self):
        print("Location:", self.location, "Delay budget:" , self.budget, "Current delay:", self.delay)

if __name__ == "__main__":
    # 1. 參數設定
    NUM_UE = 100
    NUM_SAT = 2
    TIME_SLOTS = 1000
    
    print(f"--- Simulation Start ---")
    print(f"UEs: {NUM_UE}, Satellites: {NUM_SAT}, Time Slots: {TIME_SLOTS}")

    # 2. 初始化物件
    # 建立 2 顆衛星
    sat_list = [satellite(id=i, location=[0, 0, 500]) for i in range(NUM_SAT)]
    
    # 建立 100 個 UE
    ue_list = [UE(location=[0, 0], id=i) for i in range(NUM_UE)]
    
    # 初始化 UE 的可見衛星列表 (假設全體可見)
    for ue in ue_list:
        ue.acquire_visible_sat(sat_list)

    # 3. 數據收集用的 List
    throughput_history = [] # 記錄每個 Slot 的成功數

    # 4. 主模擬迴圈
    for t in range(TIME_SLOTS):
        
        # --- Step A: 更新時間與產生封包 ---
        for ue in ue_list:
            ue.new_time()

        # --- Step B: 計算參數與決定策略 ---
        # 目前 calculate_ACB 是固定 0.5，之後要把註解打開
        for ue in ue_list:
            ue.calculate_ACB(current_time=t)
            ue.determine_order()

        # --- Step C: 執行接入測試 (ACB Test & Transmission) ---
        for ue in ue_list:
            # 如果通過 ACB，會呼叫 sat.receive_preamble()
            ue.ACB_test()

        # --- Step D: 衛星端處理 (碰撞檢測) ---
        total_success_ids_in_this_slot = []
        for sat in sat_list:
            # 回傳該衛星成功接收的 UE ID 列表
            successes = sat.check_RA_success()
            total_success_ids_in_this_slot.extend(successes)

        # 記錄本時間點的總吞吐量
        throughput_history.append(len(total_success_ids_in_this_slot))

        # --- Step E: 回傳結果給 UE (更新狀態) ---
        for ue in ue_list:
            ue.receive_feedback(total_success_ids_in_this_slot)

    # 5. 統計結果
    total_success_packets = sum(throughput_history)
    total_lost_packets = sum(ue.loss for ue in ue_list)
    avg_throughput = total_success_packets / TIME_SLOTS
    successesful_rate = total_success_packets/(total_success_packets+total_lost_packets)

    print(f"\n--- Simulation Complete ---")
    print(f"Total Successful Accesses: {total_success_packets}")
    print(f"Total Dropped Packets: {total_lost_packets}")
    print(f"Average Throughput (packets/slot): {avg_throughput:.2f}")
    print(f"Successful rate: {successesful_rate}")


    if total_success_packets > 0:
        print(">> System validation: PASSED (Traffic is flowing)")
    else:
        print(">> System validation: WARNING (No success recorded, check active_prob or Z)")