import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
import orbit
from datetime import datetime, timezone, timedelta  # 必須有 timedelta
import Load_estimator

class controller:
    def __init__(self):
        self.satellites = []
        self.sat_num = len(self.satellites) 
        self.Dmax = 5 #Delay budget的最大值
        self.p_b = np.zeros(self.Dmax)  
        self.S = []
    def add_satellite(self, satellite):
        self.satellites.append(satellite)
        self.sat_num = len(self.satellites) 
        self.S = np.ones(self.sat_num)
    def load_estimator(self, expected_tables):
        #取得衛星回報的 N_i, N_s, N_c
        N_i = np.zeros(self.sat_num)
        N_s = np.zeros(self.sat_num)
        N_c = np.zeros(self.sat_num)
        for i, sat in enumerate(self.satellites):
            N_i[i], N_s[i], N_c[i] = sat.report()
        #實作MoM estimator
        Lambda = Load_estimator.load_estimator(N_i, N_s, N_c, tables=expected_tables) #傳入預計算好的期望值表
        return Lambda
    def backoff_control(self, total_load):
        #實作backoff control
        self.p_b = np.zeros(self.Dmax)  #temp
        return 
    def satellite_selection(self,Lambda):
        self.S = np.ones(self.sat_num) #temp
        return 

class satellite:
    def __init__(self, id, skyfield_sat, Z=2):
        self.id = id
        self.skyfield_sat = skyfield_sat
        self.Z = Z          # Number of available preambles 固定為54
        self.ue_pre = {} # 暫存這個 Time Slot 嘗試接入的 UE
        self.N_i = 0 # Idle preambles
        self.N_s = 0 # Successful preambles 
        self.N_c = 0 # Collided preambles
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
        self.N_s = len(success_list)
        self.N_c = len(duplicates)
        self.N_i = self.Z - self.N_s - self.N_c
        return success_list

    def report(self):
        return self.N_i, self.N_s, self.N_c

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
        self.QoS_requirement = [0.1,0.2,0.3,0.2,0.2] #對應不同delay budget的QoS需求，總和為1
        self.visible_satellites = []
        self.geo = wgs84.latlon(self.location[0], self.location[1])
    def acquire_visible_sat(self,sat_list,current_time_obj):
        self.visible_satellites = []
        for sat in sat_list:
            if is_visible(self.geo, sat.skyfield_sat, min_elevation=20, t=current_time_obj):
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
        r = np.random.rand()
        self.budget = np.random.choice([1, 2, 3, 4, 5], p=self.QoS_requirement) #根據QoS需求隨機分配delay budget
        #self.budget = 5 #其實這有點像是NACK重傳的次數限制，季報就先這樣講吧
        self.delay = 0
        self.target_satellite = None
    def acquire_SIB(self,ctrl):
        #取得系統資訊，包含backoff機率和衛星選擇資訊
        self.p_b = ctrl.p_b
        self.S = ctrl.S
    def ACB_test(self):
        backoff = False
        target_sat = None
        r= np.random.rand()
        if r < self.p_b[self.budget-self.delay-1]: # Backoff
            backoff = True
        if not backoff:
            # 取得目前可見衛星在全體衛星集合中的 ID 
            visible_ids = [sat.id for sat in self.visible_satellites]
            # 提取對應的分數並計算機率 a_{i,k}
            scores = np.array([self.S[k] for k in visible_ids])
            exp_S = np.exp(scores)
            prob_S = exp_S / np.sum(exp_S)            
            # 隨機選擇一顆衛星並執行 RA
            chosen_idx = np.random.choice(len(self.visible_satellites), p=prob_S)
            target_sat = self.visible_satellites[chosen_idx]
            
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

def main(NUM_UE, NUM_SAT, SECONDS, MODE, SEED):
    # 模式設定
    np.random.seed(SEED) # 固定隨機種子以確保可重現性

    print(f"--- Simulation Start ---")
    print(f"UEs: {NUM_UE}, Satellite orbits: {NUM_SAT}, Time Slots: {SECONDS}")
    # 取得真實衛星列表 (Top-N 軌道面)
    # 設定模擬開始時間
    ts = load.timescale()
    start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc)
    t_start = ts.from_datetime(start_dt)
    
    # 設定觀察點 (台北)
    geo = wgs84.latlon(25.03, 121.56)

    #print("Load satellite information...")
    # 呼叫 Satellite.py 的函式 
    real_sats = orbit.get_relevant_rail_planes(t_start, geo, top_n=NUM_SAT)
    
    #設定controller
    ctrl = controller()

    # 將真實衛星「封裝」進您的 Simulation Class
    sat_list = []
    for i, real_sat in enumerate(real_sats):
        s = satellite(id=i, skyfield_sat=real_sat)
        sat_list.append(s)
        ctrl.add_satellite(s)
    
    #print(f"Complete building environment: {len(sat_list)} satellites loaded.")

    ue_list = []
    for i in range(NUM_UE):
        lat = 25.03 + np.random.uniform(-1, 1)
        lon = 121.56 + np.random.uniform(-1, 1)
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
    expected_tables = Load_estimator.precompute_expected_tables(Z=sat_list[0].Z, Nmax=1000) #預計算期望值表，傳入Z值和Nmax上限
    # 主模擬迴圈
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數
    for n in range(RAO_COUNTS): #統一用n，表示現在是在第幾個RAO
        # --- 更新時間與產生封包 ---
        burst_idx = n*trao // tb #計算目前在哪個Burst period
        for ue in ue_list:
            burst_idx_ue = burst_idx*tb//trao + arrival_rao[burst_idx * NUM_UE + ue.id] #每個UE在每個Burst period都會有一個對應的bursty time
            ue.new_time(bursty=True if burst_idx_ue == n else False) #如果目前的Slot時間超過了該UE的bursty time，則產生新封包
        # --- 衛星移動與可見衛星列表更新 ---
        if n % 5 == 0 or n == 1: #每5個RAO更新一次可見衛星列表，因為衛星移動不會太快
            current_ms = n * trao
            current_dt = start_dt + timedelta(milliseconds=current_ms)
            current_t = ts.from_datetime(current_dt)
            for ue in ue_list:
                ue.acquire_visible_sat(sat_list, current_t)
        
        #Controller-side processing
        Lambda = ctrl.load_estimator(expected_tables) #每個RAO都呼叫一次load estimator，並且傳入預計算好的期望值表
        ctrl.backoff_control(total_load=sum(Lambda)) 
        ctrl.satellite_selection(Lambda=Lambda) 

        for ue in ue_list:
            if ue.active: #只對active的UE計算ACB和決定順序
                ue.acquire_SIB(ctrl)

        # UE-side processing
        for ue in ue_list:
            # 如果通過 ACB，會呼叫 sat.receive_preamble()
            if ue.active: 
                ue.ACB_test()

        # [新增] 進度條與監控資訊 (每 50 slots 印一次)
        if n % 50 == 0:
            # 計算當前統計數據
            active_count = sum(u.active for u in ue_list)
            # 計算平均可視衛星數
            avg_vis_sats = np.mean([len(u.visible_satellites) for u in ue_list])
            # 使用 \r 讓同一行刷新，不會洗版
            print(f"Slot {n}/{RAO_COUNTS} | Active: {active_count:3d} | AvgVisSat: {avg_vis_sats:.1f}", end='\r')
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
            if ue.active: #只有active的UE才會收到反饋，並且可能改變狀態
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