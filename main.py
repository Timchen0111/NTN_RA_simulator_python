import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timezone
import orbit

class satellite:
    def __init__(self, id, skyfield_sat, Z=54):
        self.id = id
        self.skyfield_sat = skyfield_sat
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
    def acquire_visible_sat(self,sat_list,current_time_obj):
        self.visible_satellites = []
        ue_geo = wgs84.latlon(self.location[0], self.location[1])
        for sat in sat_list:
            if is_visible(ue_geo, sat.skyfield_sat, min_elevation=1, t=current_time_obj):
                self.visible_satellites.append(sat)
    def new_time(self,bursty):
        if self.active == True:
            self.delay += 1
            if self.delay > self.budget:
                self.active = False
                self.loss += 1
                self.delay = 0
        else:
            if bursty:
                self.new_packet()
    def new_packet(self):
        self.active = True
        self.budget = np.random.randint(5) #改成數個不同類型的 UE，分別有不同的 delay budget
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
        self.order = np.random.permutation(len(self.visible_satellites)) #純隨機，之後再改 可以先寫比ranking score的排序機制，至於ranking score值先隨便定
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
    def Traditional_ACB_test(self):
        if not self.active or self.visible_satellites == []:
            return
        chosen_sat_idx = np.random.choice(len(self.visible_satellites))
        p_acb = self.ACB[chosen_sat_idx] #固定值
        if np.random.rand() < p_acb:
            # 通過 ACB，隨機選擇一顆衛星進行接入
            target_sat = self.visible_satellites[chosen_sat_idx]
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

def is_visible(UE_location, satellite, min_elevation,t):
    difference = satellite - UE_location
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > min_elevation

def main(NUM_UE, NUM_SAT, SECONDS, MODE):
    # 模式設定
    if MODE == 1:
        SBC = True  
    else:
        SBC = False

    print(f"--- Simulation Start ---")
    print(f"UEs: {NUM_UE}, Satellites: {NUM_SAT}, Time Slots: {SECONDS}")
    # 取得真實衛星列表 (Top-N 軌道面)
    # 設定模擬開始時間
    ts = load.timescale()
    start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc)
    t_start = ts.from_datetime(start_dt)
    
    # 設定觀察點 (台北)
    taipei_geo = wgs84.latlon(25.03, 121.56)

    print("Load satellite information...")
    # 呼叫 Satellite.py 的函式 
    real_sats = orbit.get_relevant_rail_planes(t_start, taipei_geo, top_n=NUM_SAT)
    
    # 將真實衛星「封裝」進您的 Simulation Class
    sat_list = []
    for i, real_sat in enumerate(real_sats):
        s = satellite(id=i, skyfield_sat=real_sat)
        sat_list.append(s)
    
    print(f"Complete building environment: {len(sat_list)} satellites loaded.")

    ue_list = []
    for i in range(NUM_UE):
        lat = 25.03 + np.random.uniform(-0.1, 0.1)
        lon = 121.56 + np.random.uniform(-0.1, 0.1)
        ue_list.append(UE(location=[lat, lon], id=i))

    # 建立Burst time table
    tb = 10000
    trao = 640
    burst_count = SECONDS*1000//tb
    samples = np.random.beta(3, 4, burst_count * NUM_UE) #每個UE在每次Burst period都會產生一個bursty time
    arrival_times = samples * tb  #將樣本轉換成毫秒，並且分布在每個Burst period的0到tb秒之間
    arrival_rao  = arrival_times//trao #離散化，算出burst在第幾個RAO index
    # 數據收集用的 List
    throughput_history = [] # 記錄每個 Slot 的成功數

    # 主模擬迴圈
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數
    for n in range(RAO_COUNTS): #統一用n，表示現在是在第幾個RAO
        # --- 更新時間與產生封包 ---
        burst_idx = n*trao // tb #計算目前在哪個Burst period
        for ue in ue_list:
            burst_idx_ue = burst_idx*tb//trao + arrival_rao[burst_idx * NUM_UE + ue.id] #每個UE在每個Burst period都會有一個對應的bursty time
            ue.new_time(bursty=True if burst_idx_ue == n else False) #如果目前的Slot時間超過了該UE的bursty time，則產生新封包
        # --- 衛星移動與可見衛星列表更新 ---
        for sat in sat_list:
            pass
        for ue in ue_list:
            ue.acquire_visible_sat(sat_list, t_start + n*trao/1000) #更新每個UE的可見衛星列表，時間要換算成秒
        # --- 計算參數與決定策略 ---
        # 目前 calculate_ACB 是固定 0.5，之後要把註解打開
        for ue in ue_list:
            ue.calculate_ACB(current_time=n)
            ue.determine_order()

        # --- SBC procedure---
        for ue in ue_list:
            # 如果通過 ACB，會呼叫 sat.receive_preamble()
            if SBC:
                ue.ACB_test()
            else:
                ue.Traditional_ACB_test()

        # --- 衛星端處理 (碰撞檢測) ---
        total_success_ids_in_this_slot = []
        for sat in sat_list:
            # 回傳該衛星成功接收的 UE ID 列表
            successes = sat.check_RA_success()
            total_success_ids_in_this_slot.extend(successes)

        # 記錄本時間點的總吞吐量
        throughput_history.append(len(total_success_ids_in_this_slot))

        # --- 回傳結果給 UE (更新狀態) ---
        for ue in ue_list:
            ue.receive_feedback(total_success_ids_in_this_slot)

    # 統計結果
    total_success_packets = sum(throughput_history)
    total_lost_packets = sum(ue.loss for ue in ue_list)
    avg_throughput = total_success_packets / (RAO_COUNTS * trao / 1000)  # packets per second
    successesful_rate = total_success_packets/(total_success_packets+total_lost_packets)

    print(f"\n--- Simulation Complete ---")
    print(f"Total Successful Accesses: {total_success_packets}")
    print(f"Total Dropped Packets: {total_lost_packets}")
    print(f"Average Throughput (packets/second): {avg_throughput:.2f}")
    print(f"Successful rate: {successesful_rate}")
    return avg_throughput, successesful_rate