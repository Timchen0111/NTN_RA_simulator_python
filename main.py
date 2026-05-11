import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
import orbit
from datetime import datetime, timezone, timedelta  # 必須有 timedelta
import Load_estimator, backoff_control, N_estimate, selection

class controller:
    def __init__(self):
        self.satellites = []
        self.sat_num = len(self.satellites) 
        self.Dmax = 5 #Delay budget的最大值
        self.p_b = np.zeros(self.Dmax)
        self.observe_pi = np.ones(self.Dmax) / self.Dmax
        self.S = []
        self.N_estimate = 0
        self.rls = N_estimate.RLSEstimator(initial_N=self.N_estimate)
        self.actualPi = np.zeros(self.Dmax) #用來記錄每個pi的真實值，供測試參考
        self.actual = []
        self.history_reward = []
    def set_agent(self):
        self.agent = selection.SatelliteSelectionAgent(satellite_list=self.satellites, S_max=5, mem_length=5)
        self.last_state = None
        self.last_action_idx = None
        self.S = np.ones(len(self.satellites))
    def add_satellite(self, satellite):
        self.satellites.append(satellite)
        self.sat_num = len(self.satellites) 
    def load_estimator(self, expected_tables):
        #取得衛星回報的 N_i, N_s, N_c
        N_i = np.zeros(self.sat_num)
        N_s = np.zeros(self.sat_num)
        N_c = np.zeros(self.sat_num)
        actual_lambda = np.zeros(self.sat_num) #用來記錄每顆衛星的真實附載，供測試參考
        for i, sat in enumerate(self.satellites):
            N_i[i], N_s[i], N_c[i], actual_lambda[i] = sat.report()
        #實作MoM estimator
        Lambda = Load_estimator.load_estimator(N_i, N_s, N_c, tables=expected_tables) #傳入預計算好的期望值表
        #Lambda = actual_lambda #測試用
        self.actualLambda = sum(actual_lambda)
        #print(f"Total load estimation: Lambda={sum(Lambda)}, Actual Lambda={sum(actual_lambda)}")
        return Lambda
    def backoff_control(self, total_load, rho, p_d, K, Z, MODE,n):
        #實作backoff control
        denominator = np.sum(self.observe_pi * (1 - self.p_b))
        N_tilde = self.N_estimation(Lambda=total_load, denominator=denominator)
        self.N_estimate = N_tilde
        #N_tilde = 10000 #丟真值測試用
        self.p_b, self.observe_pi = backoff_control.backoff_control(N_tilde, self.p_b, rho, self.Dmax, p_d, K, Z,MODE)
        if n % 10 == 0:
            #print(f"Actual Pi: {self.actualPi}, Observed Pi: {self.observe_pi}")
            self.actual.append(list(self.actualPi[:5]))
        #print(f"Backoff control updated: p_b={self.p_b}, pi={self.observe_pi}")
        return
    def baseline_score(self, Lambda,n):
        #實作簡單的基於負載的衛星選擇，分數與負載成反比
        # 1. 取得負載的倒數（與負載成反比）
        eps = 10 #Avoid 0 exploration
        inv_lambda = 1.0 / (Lambda + eps)
        # 2. 進行歸一化與縮放
        # 將倒數關係映射到 [1, S_max] 區間
        # 邏輯：負載最低的衛星拿 S_max，其餘依比例分配
        max_inv = np.max(inv_lambda)
        if max_inv > 0:
            S = (inv_lambda / max_inv) * self.agent.S_max
            # 3. 離散化並確保最小值為 1
        S = np.round(np.clip(S, 1, self.agent.S_max))
        return S 
    def satellite_selection(self, Lambda,MODE,n,target_location,t):
        reward = self.agent.compute_reward(Lambda)
        self.history_reward.append(reward)
        if MODE == 2 or MODE == 3:
            if n % 50 == 0 and n>0:
                #print(f"Lambda:{Lambda}")
                print(f"Current Satellite Preference Score: {self.S}")
                print(f"Current Reward: {reward}")
            return # MODE 2 和 MODE 3 不執行衛星選擇，S固定為1，也就是隨機選擇
        if MODE == 4:
            self.S = self.baseline_score(Lambda,n) # MODE 4 直接使用啟發式分數，不進行 RL 微調
            if n % 50 == 0 and n>0:
                print(f"Current Satellite Preference Score: {self.S}")
                print(f"Current Reward: {reward}")
            return
        # 1. 取得目前狀態 s_m (對應 H^m-1)
        angle = np.zeros(self.sat_num)
        if MODE == 1:
            baseline_score = None
        else:
            baseline_score = self.baseline_score(Lambda,n) #mode 5: reward shaping，讓 RL 主要學習微調分數而非從零開始
        for i in range(self.sat_num):
            angle[i] = self.satellites[i].calculate_elevation_angle(target_location, t)
        current_state = self.agent.get_state(Lambda,angle)
        # 2. 學習階段：利用「當下觀測到的 Lambda」來評價「上一回合做的決定」
        if self.last_state is not None:
            # 計算上一回合動作 a_{m-1} 產生的獎勵 r_{m-1}
            reward = self.agent.compute_reward(Lambda)
            # 將 (s_{m-1}, a_{m-1}, r_{m-1}, s_m) 存入經驗池
            self.agent.store_transition(
                self.last_state, 
                self.last_action_idx, 
                reward, 
                current_state
            )
            # 觸發神經網路優化 (update theta)
            self.agent.update_policy()
        # 3. 決策階段：決定這一回合要使用的動作 a_m 與分數向量 S_m
        # 這裡會用到 epsilon-greedy，產出 action_idx 用於未來學習
        now_ep = max(0.05, 0.5 - 0.005*n) #linearly 隨時間衰減 epsilon，從0.5開始，每20個RAO減少0.1，最低到0.05
        action_idx, current_S = self.agent.select_action(current_state, epsilon=now_ep, S_heuristic=baseline_score)
        # 4. 紀錄目前的資訊，供下一輪 (m+1) 學習使用
        self.last_state = current_state
        self.last_action_idx = action_idx
        self.S = current_S
        if len(self.S) != self.sat_num:
            raise ValueError(f"Warning: S length {len(self.S)} does not match number of satellites {self.sat_num}.")
        if n % 50 == 0 and n>0:
            print(f"Current Satellite Preference Score: {self.S}")
            print(f"Current Reward: {reward}")
        return
    def N_estimation(self, Lambda, denominator):
        current_N = self.rls.update(Lambda, denominator)
        #current_N = Lambda/denominator if denominator > 0 else self.N_estimate #測試用
        return current_N
    
class satellite:
    def __init__(self, id, skyfield_sat, Z=54):
        self.id = id
        self.skyfield_sat = skyfield_sat
        self.Z = Z          # Number of available preambles 固定為54
        self.ue_pre = {} # 暫存這個 Time Slot 嘗試接入的 UE
        self.N_i = 0 # Idle preambles
        self.N_s = 0 # Successful preambles 
        self.N_c = 0 # Collided preambles
        self.actual_lambda = 0 # 真實附載 (UE數量)，供測試參考
    def assign_id(self, new_id):
        self.id = new_id
    def receive_preamble(self,ue_id):
        # 模擬 UE隨機選取一個 Preamble (0 到 Z-1)
        chosen_preamble = np.random.randint(0, self.Z)
        self.ue_pre[ue_id] = chosen_preamble
    def calculate_elevation_angle(self,target_location,t):
        difference = self.skyfield_sat - target_location
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        return alt.degrees
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
        self.actual_lambda = len(self.ue_pre) #記錄真實附載供測試參考
        self.ue_pre.clear()
        self.N_s = len(success_list)
        self.N_c = len(duplicates)
        self.N_i = self.Z - self.N_s - self.N_c
        return success_list

    def report(self):
        return self.N_i, self.N_s, self.N_c, self.actual_lambda

    def get_info(self):
        print("ID:", self.id, "Location:", self.location)

class UE:
    def __init__(self,location,id,rho):
        self.id = id
        self.location = location
        self.budget = 0
        self.delay = 0
        self.loss = 0
        self.success = 0
        self.active = False #Boolean
        self.active_prob = rho
        self.QoS_requirement = [0.2,0.2,0.2,0.2,0.2] #對應不同delay budget的QoS需求，總和為1
        self.visible_satellites = []
        self.geo = wgs84.latlon(self.location[0], self.location[1])
    def acquire_visible_sat(self,sat_list,current_time_obj):
        self.visible_satellites = []
        #counter = 1
        for sat in sat_list:
            if is_visible(self.geo, sat.skyfield_sat, min_elevation=20, t=current_time_obj): #所以這行是Bottleneck，當UE數量大就會變很慢。
                self.visible_satellites.append(sat) #測試版，假設都只能看到衛星1-5，這強迫UCP assumption成立，作為固定變因
            #counter += 1
            #if counter > 10:
             #  break

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
        if not backoff and len(self.visible_satellites) > 0:
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

def evaluate_visibility_heterogeneity(ue_list, num_samples=100):
    """
    評估 UE 可見衛星集合的異質性
    :param ue_list: 當前時隙的所有 UE 列表
    :param num_samples: 隨機採樣的對數，避免全量計算（O(N^2)）導致效能崩潰
    """
    if len(ue_list) < 2:
        return {"jaccard": 1.0, "unique_ratio": 0.0, "cv": 0.0}

    # 1. 計算 Jaccard Similarity (量化重疊度)
    jaccard_indices = []
    # 限制採樣數以提升速度
    pairs = [np.random.choice(ue_list, 2, replace=False) for _ in range(min(num_samples, len(ue_list)//2))]
    
    for u1, u2 in pairs:
        set1 = set(s.id for s in u1.visible_satellites)
        set2 = set(s.id for s in u2.visible_satellites)
        
        union = set1.union(set2)
        if len(union) == 0:
            continue
        
        intersection = set1.intersection(set2)
        jaccard_indices.append(len(intersection) / len(union))
    
    avg_jaccard = np.mean(jaccard_indices) if jaccard_indices else 1.0

    # 2. 統計每顆衛星被看見的次數 (量化空間壓力分佈)
    # 假設所有可能的衛星 ID 已經在 active_sat_pool 中
    sat_appearance = {}
    for ue in ue_list:
        for sat in ue.visible_satellites:
            sat_appearance[sat.id] = sat_appearance.get(sat.id, 0) + 1
    
    # 計算變異係數 (CV)
    counts = list(sat_appearance.values())
    cv = np.std(counts) / np.mean(counts) if counts else 0.0
    
    # 3. 統計全體 UE 覆蓋的獨特衛星總數
    unique_sats = len(sat_appearance.keys())

    return {
        "avg_jaccard": avg_jaccard, # 越小代表 UE 看到的星越不一樣 (RL 更有利)
        "cv_visibility": cv,       # 越大代表衛星間壓力越不均 (選星越重要)
        "total_unique_sats": unique_sats # 系統當前總共利用了幾顆衛星
    }

def main(RHO, NUM_SAT, SECONDS, NUM_UE,MODE, SEED):
    # 模式設定
    np.random.seed(SEED) # 固定隨機種子以確保可重現性

    print(f"--- Simulation Start ---")
    print(f"Active rate: {RHO}, Satellite orbits: {NUM_SAT}, Time Slots: {SECONDS}")
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
        s = satellite(id=0, skyfield_sat=real_sat) #這裡不賦予id，因為後面會篩選可見衛星池，id會對不上，所以直接在satellite class裡面用list index當id。
        sat_list.append(s)
        
    
    #print(f"Complete building environment: {len(sat_list)} satellites loaded.")

    ue_list = []
    '''
    for i in range(NUM_UE):
        lat = 25.03 + np.random.uniform(-1, 1)
        lon = 121.56 + np.random.uniform(-1, 1)
        ue_list.append(UE(location=[lat, lon], id=i, rho=RHO))
    
    centers = [
        [25.03, 121.56], 
        [24.14, 120.67]
    ]
    '''
    for i in range(NUM_UE):
        # 隨機決定該 UE 屬於哪一個熱點群體 (各占 50%) 暫時關掉
        c = [25.03, 121.56] #centers[np.random.choice([0, 1])]
        
        # 使用立方變換製造聚集效應
        # 我們將偏移量限制在 +-1.5 之間，再取立方以強化核心密集度
        raw_lat_offset = np.random.uniform(-2, 2)
        raw_lon_offset = np.random.uniform(-2, 2)
        
        # 立方變換：(offset^3) / 2.25 確保範圍大致維持在原有尺度，但極度集中
        lat = c[0] + (raw_lat_offset ** 3) / 2.25
        lon = c[1] + (raw_lon_offset ** 3) / 2.25
        
        ue_list.append(UE(location=[lat, lon], id=i, rho=RHO))
        
    # 建立Burst time table
    #tb = 10000
    trao = 100
    #burst_count = SECONDS*1000//tb
    #samples = np.random.beta(3, 4, burst_count * NUM_UE) #每個UE在每次Burst period都會產生一個bursty time
    #arrival_times = samples * tb  #將樣本轉換成毫秒，並且分布在每個Burst period的0到tb秒之間
    #arrival_rao  = arrival_times//trao #離散化，算出burst在第幾個RAO index
    # 數據收集用的 List
    throughput_history = [] # 記錄每個 Slot 的成功數
    n_history = [] # 記錄每個 Slot 的 N_estimate
    expected_tables = Load_estimator.precompute_expected_tables(Z=sat_list[0].Z, Nmax=1000) #預計算期望值表，傳入Z值和Nmax上限
    # 主模擬迴圈
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數
    active_sat_pool = []
    for sat in sat_list:
    #暫時用區域中心點的衛星仰角篩選一部份的衛星，允許的仰角閾值更加小因為只是初步篩選。
        mid_dt = start_dt + timedelta(seconds=SECONDS/2)
        if is_visible(geo, sat.skyfield_sat, min_elevation=10, t=ts.from_datetime(mid_dt)): #所以這行是Bottleneck，當UE數量大就會變很慢。改成模擬時間中點，模擬日昇日落而非一直日落
            active_sat_pool.append(sat) #這些就是模擬中我們系統考慮的衛星池，後續不再更動。
    print(f"Active Sat Pool Size: {len(active_sat_pool)}")
 
    for i in range(len(active_sat_pool)):
        sat = active_sat_pool[i]
        ctrl.add_satellite(sat) #Controller只加入active_sat_pool裡的衛星
        sat.assign_id(i) #為每個衛星分配新的ID
    
    ctrl.set_agent() #在加入衛星後初始化Agent，讓Agent知道目前的衛星列表和數量

    for n in range(RAO_COUNTS): #統一用n，表示現在是在第幾個RAO
        # --- 更新時間與產生封包 ---
        '''
        burst_idx = n*trao // tb #計算目前在哪個Burst period
        for ue in ue_list:
            burst_idx_ue = burst_idx*tb//trao + arrival_rao[burst_idx * NUM_UE + ue.id] #每個UE在每個Burst period都會有一個對應的bursty time
            ue.new_time(bursty=True if burst_idx_ue == n else False) #如果目前的Slot時間超過了該UE的bursty time，則產生新封包
        '''
        for ue in ue_list:
            is_new_packet = np.random.rand() < (RHO * 1000 / trao)
            ue.new_time(bursty=is_new_packet)
        #'''

        current_ms = n * trao
        current_dt = start_dt + timedelta(milliseconds=current_ms)
        current_t = ts.from_datetime(current_dt)
        # --- 衛星移動與可見衛星列表更新 ---
        if n % 5 == 0 or n == 1: #每5個RAO更新一次可見衛星列表，因為衛星移動不會太快
            visible_count = 0
            sat_visibility_counts = np.zeros(len(active_sat_pool))
            for ue in ue_list:
                ue.acquire_visible_sat(active_sat_pool, current_t)
                visible_count += len(ue.visible_satellites)
                for sat in ue.visible_satellites:
                    sat_visibility_counts[sat.id] += 1
            # 計算變異係數 (CV)，衡量衛星間潛在競爭壓力的不均勻性
            cv_visibility = np.std(sat_visibility_counts) / np.mean(sat_visibility_counts)
            avg_visible = visible_count / NUM_UE
            if n % 50 == 0 and n>0:
                print(f"RAO {n}: Average visible satellites per UE: {avg_visible:.2f}")
                print(f"RAO {n}: Coefficient of Variation for Satellite Visibility: {cv_visibility:.2f}")
            if avg_visible < 1:
                print("Warning: Too few visible satellites on average. The simulation scenario is not feasible. Ending simulation.")
                return
            if MODE == 0: #測試模式，不是真的跑模擬
                eval_metrics = evaluate_visibility_heterogeneity(ue_list)
                return eval_metrics
        real_counts = np.zeros(5)
        idle_ue_count=0
        for ue in ue_list:
            if ue.active:
                # 取得該 UE 剩餘的延遲預算 
                nn = ue.budget - ue.delay
                if nn > 0:
                    real_counts[nn-1] += 1
            else:
                idle_ue_count += 1
        
        #real_counts[0] = idle_ue_count
        ctrl.actualPi = real_counts / NUM_UE #更新真實pi供測試參考
        #Controller-side processing
        Lambda = ctrl.load_estimator(expected_tables) #每個RAO都呼叫一次load estimator，並且傳入預計算好的期望值表
        #if n % 10 == 0: #減少更新backoff control的頻率，讓系統有機會達到所需的穩態假設
        ctrl.backoff_control(total_load=sum(Lambda), rho=(RHO * 1000 / trao), p_d = ue_list[0].QoS_requirement, K=ctrl.sat_num, Z=sat_list[0].Z,MODE=MODE,n=n)
        ctrl.satellite_selection(Lambda=Lambda,MODE=MODE, n=n, target_location=geo, t=current_t)
        current_n_hat = ctrl.N_estimate
        n_history.append(current_n_hat)
        print("Successfully completed one round of controller processing.")
        
        if n % 50 == 0:
            print(f"Current N_tilde: {ctrl.N_estimate}, Total Load (Lambda): {sum(Lambda)}, Backoff rate: {ctrl.p_b}", end='\n')

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
    return avg_throughput, successesful_rate, n_history, ctrl.actual, ctrl.observe_pi, ctrl.history_reward
