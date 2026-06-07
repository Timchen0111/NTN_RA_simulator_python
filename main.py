import numpy as np
from skyfield.api import load, wgs84
import orbit
from datetime import datetime, timezone, timedelta  # 必須有 timedelta
import Load_estimator, backoff_control, N_estimate, selection
import json

class controller:
    def __init__(self, group_weight_table=None, group_ps_table=None):
        self.satellites = []
        self.sat_num = len(self.satellites) 
        self.Dmax = 5 #Delay budget的最大值
        self.p_b = np.zeros(self.Dmax)
        self.observe_pi = np.ones(self.Dmax) / self.Dmax
        self.group_weight_table = group_weight_table
        self.group_ps_table = group_ps_table
        self.A_by_group = {}
        self.N_estimate = 0
        self.rls = N_estimate.RLSEstimator(initial_N=self.N_estimate)
        self.actualPi = np.zeros(self.Dmax) #用來記錄每個pi的真實值，供測試參考
        self.actual = []
        self.history_reward = []
        self.ue_list = []
        self.selection_solver_failures = 0
    def set_group_probabilities_for_rao(self, n, use_convex_solver=False, imbalance_epsilon=0.01):
        if self.group_weight_table is None:
            self.A_by_group = {}
            return
        weights = self.group_weight_table[n]
        ps_by_group = self.group_ps_table[n] if self.group_ps_table is not None else None
        if use_convex_solver and ps_by_group is not None:
            try:
                self.A_by_group = selection.solve_group_selection_policy(
                    weights=weights,
                    ps_by_group=ps_by_group,
                    sat_num=self.sat_num,
                    imbalance_epsilon=imbalance_epsilon,
                    initial_policy=self.A_by_group if self.A_by_group else None,
                )
                return
            except Exception as exc:
                self.selection_solver_failures += 1
                print(
                    f"Warning: convex satellite selection failed at RAO {n}; "
                    f"falling back to uniform A_g. Reason: {exc}"
                )

        current_probabilities = {}
        for group in weights.keys():
            group_key = tuple(group)
            if self.sat_num <= 0:
                raise ValueError("Cannot build A_g before satellites are registered.")
            current_probabilities[group_key] = np.ones(self.sat_num) / self.sat_num
        self.A_by_group = current_probabilities
    def set_agent(self):
        self.last_state = None
        self.last_action_idx = None
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
    def backoff_control(self, total_load, rho, p_d, p_s, K, Z, MODE,n):
        #實作backoff control
        denominator = np.sum(self.observe_pi * (1 - self.p_b))
        N_tilde = self.N_estimation(Lambda=total_load, denominator=denominator)
        self.N_estimate = N_tilde
        #N_tilde = 10000 #丟真值測試用
        self.p_b, self.observe_pi = backoff_control.backoff_control(N_tilde, self.p_b, rho, self.Dmax, p_d, p_s, K, Z, MODE, total_load)
        if n % 10 == 0:
            #print(f"Actual Pi: {self.actualPi}, Observed Pi: {self.observe_pi}")
            self.actual.append(list(self.actualPi[:5]))
        #print(f"Backoff control updated: p_b={self.p_b}, pi={self.observe_pi}")
        return
    def satellite_selection(self, Lambda,MODE,n,target_location,t,epoch):
        avg_load = np.mean(Lambda)
        reward = -np.mean((Lambda - avg_load) ** 2)
        self.history_reward.append(reward)
        if n % 50 == 0 and n>0:
            print(f"Current group selection policies: {len(self.A_by_group)} groups")
            print(f"Current Reward: {reward}")
        return
    def N_estimation(self, Lambda, denominator):
        current_N = self.rls.update(Lambda, denominator)
        #current_N = Lambda/denominator if denominator > 0 else self.N_estimate #測試用
        return current_N
    def reset_agent(self):
        self.last_state = None
        self.last_action_idx = None
    
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
    def receive_preamble(self,ue_id,angle, distance):
        # 模擬 UE隨機選取一個 Preamble (0 到 Z-1)
        if channel_calculator(angle, distance):
            chosen_preamble = np.random.randint(0, self.Z)
            self.ue_pre[ue_id] = chosen_preamble
            return True
        else:
            return False #模擬傳輸失敗，UE不會被記錄在ue_pre裡面，controller也不會收到這個UE的任何回報，這相當於UE根本沒有嘗試接入一樣
    def calculate_elevation_angle(self,target_location,t):
        difference = self.skyfield_sat - target_location
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        return alt.degrees, distance.km
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
        self.group = None #這個UE被分配到的衛星群組，初始為None，之後會根據可見衛星列表更新
        self.transmission_success = 0 #實際成功傳輸的次數，減掉self.success後就是碰撞次數
        self.transmission_fail = 0
        self.active = False #Boolean
        self.active_prob = rho
        self.QoS_requirement = [0.2,0.2,0.2,0.2,0.2] #對應不同delay budget的QoS需求，總和為1
        self.visible_satellites = []
        self.all_satellites = []
        self.A_g = None
        self.acb_selection_count = 0
        self.acb_policy_fallback_count = 0
        self.geo = wgs84.latlon(self.location[0], self.location[1])
    def acquire_visible_sat(self,sat_list,current_time_obj,mode,num):
        self.visible_satellites = []
        self.all_satellites = list(sat_list)
        self.angle = np.zeros(len(sat_list))        
        self.distance = np.zeros(len(sat_list))
        if mode == 6 or mode == 7:
            for sat in sat_list:
                self.visible_satellites.append(sat) #全部都看的到
        else:
            index = 0
            for sat in sat_list:
                visible, angle, distance = channel_visibility(self.geo, sat.skyfield_sat, min_elevation=0, t=current_time_obj)
                self.angle[index] = angle
                self.distance[index] = distance
                # 中文註解：對齊 preselection，仰角小於等於 0 的衛星不放入 UE 本輪可選 active set。
                if visible:
                    self.visible_satellites.append(sat)
                index += 1
            sorted_indices = np.argsort(self.angle)[::-1]
            # 提取前兩好衛星的實體 ID 或是物件指標
            k1_sat_id = sat_list[sorted_indices[0]].id
            k2_sat_id = sat_list[sorted_indices[1]].id
            # 將群組定義為有序雙星序對 (Ordered Pair)
            # 這樣能精確捕捉 UE 位於 Cell 哪一側的視界不對稱特性
            self.group = (k1_sat_id, k2_sat_id)

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
        #print(f"UE {self.id}, group {self.group}")
        self.p_b = ctrl.p_b
        if self.group is None:
            self.A_g = None
            return
        group_key = tuple(self.group)
        if group_key not in ctrl.A_by_group:
            self.A_g = None
            return
        group_probabilities = ctrl.A_by_group[group_key]
        if len(group_probabilities) != ctrl.sat_num:
            raise ValueError(f"UE {self.id} received A_g length {len(group_probabilities)}, expected {ctrl.sat_num}.")
        self.A_g = group_probabilities
    def ACB_test(self):
        backoff = False
        target_sat = None
        r= np.random.rand()
        if r < self.p_b[self.budget-self.delay-1]: # Backoff
            backoff = True
        candidate_satellites = self.all_satellites if len(self.all_satellites) > 0 else self.visible_satellites
        if not backoff and len(candidate_satellites) > 0:
            self.acb_selection_count += 1
            if self.A_g is None:
                self.acb_policy_fallback_count += 1
                fallback_satellites = self.visible_satellites if len(self.visible_satellites) > 0 else candidate_satellites
                target_sat = max(fallback_satellites, key=lambda sat: self.angle[sat.id])
                self.execute_RA(target_sat)
                return
            candidate_ids = [sat.id for sat in candidate_satellites]
            probabilities = np.array([self.A_g[k] for k in candidate_ids], dtype=float)
            prob_sum = np.sum(probabilities)
            if prob_sum <= 0 or not np.isfinite(prob_sum):
                self.acb_policy_fallback_count += 1
                fallback_satellites = self.visible_satellites if len(self.visible_satellites) > 0 else candidate_satellites
                target_sat = max(fallback_satellites, key=lambda sat: self.angle[sat.id])
                self.execute_RA(target_sat)
                return
            probabilities = probabilities / prob_sum
            # 隨機選擇一顆衛星並執行 RA
            chosen_idx = np.random.choice(len(candidate_satellites), p=probabilities)
            target_sat = candidate_satellites[chosen_idx]            
            self.execute_RA(target_sat)
        else:
            # Backoff: 本回合不傳輸
            pass

    def execute_RA(self,target_sat):
        #實際傳輸 Preamble
        r = target_sat.receive_preamble(self.id, self.angle[target_sat.id], self.distance[target_sat.id])    
        if r:
            self.transmission_success += 1
        else:
            self.transmission_fail += 1
        
    def receive_feedback(self, success_list):
        #接收衛星回傳的結果並更新狀態
        if self.id in success_list:
            # 成功接入，重置狀態
            self.active = False
            self.success += 1
        else:
            # 碰撞失敗，保持 active，下回合 delay 會增加
            pass

def calculate_ps(ctrl,n,group_weight_table, group_ps_table):
    weights = group_weight_table[n]
    ps_by_group = group_ps_table[n]
    # 中文註解：依照公式 p_s = sum_g w_g sum_k a_{g,k} p_{s,k}^g 計算，p_{s,k}^g 由預計算表提供。
    p_s = 0.0
    for group, w_g in weights.items():
        group_key = tuple(group)
        ps_g = ps_by_group[group_key]   # shape: (K,)
        if len(ps_g) != ctrl.sat_num:
            raise ValueError(f"p_s vector length {len(ps_g)} does not match number of satellites {ctrl.sat_num} for group {group_key}")
        if group_key not in ctrl.A_by_group:
            raise KeyError(f"Missing A_g for group {group_key} at RAO {n}")

        a_g = np.asarray(ctrl.A_by_group[group_key], dtype=float)
        if len(a_g) != ctrl.sat_num:
            raise ValueError(f"A_g length {len(a_g)} does not match number of satellites {ctrl.sat_num} for group {group_key}")
        if np.any(a_g < 0) or not np.all(np.isfinite(a_g)):
            raise ValueError(f"A_g contains invalid probabilities for group {group_key}")
        prob_sum = np.sum(a_g)
        if prob_sum <= 0:
            continue
        a_g = a_g / prob_sum
        group_success = np.sum(a_g * ps_g)
        p_s += w_g * group_success
    return p_s

def channel_calculator(elevation_angle, distance_km):
    if elevation_angle <= 0:
        return False

    LOS_PROB = {
        # 新增 0 度 anchor，讓 0~10 度的 LOS 機率做線性插值；
        # 舊版會把 0~10 度全部 clip 成 10 度，會高估低仰角鏈路。
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "prob": [0.0, 0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998],
        #"elevation_deg": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        #"prob": np.array([0.246, 0.386, 0.493, 0.613, 0.726, 0.805, 0.919, 0.968, 0.992]),
    }
    CHANNEL_PARAMETER = {
        # 0~10 度同樣保留一個 0 度 anchor，供 np.interp 線性插值。
        # sigma 是量測/模型 fitting 的標準差，低仰角不一定單調，因此 0 度先沿用 10 度值。
        # CL 則有較明顯低仰角惡化趨勢，因此用 10/20 度趨勢外插得到 0 度值 20.87 dB。
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "los_sigma_sf_db":  [1.79, 1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72],
        "nlos_sigma_sf_db": [8.93, 8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52],
        "nlos_cl_db":       [20.87, 19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30]
        #"los_sigma_sf_db":  np.array([4, 4, 4, 4, 4, 4, 4, 4, 4]),
        #"nlos_sigma_sf_db": np.array([6, 6, 6, 6, 6, 6, 6, 6, 6]),
        #"nlos_cl_db":            np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5]),
    }
    # 只限制插值表的有效範圍；0 度以下已經在前面 return False。
    # 因此 0<angle<10 時會由 np.interp 在 0 與 10 度 anchor 之間線性插值。
    elevation_angle = np.clip(elevation_angle, 0, 90)

    p_los = np.interp(
        elevation_angle,
        LOS_PROB["elevation_deg"],
        LOS_PROB["prob"]
    )

    is_los = np.random.rand() < p_los

    if is_los:
        sigma_sf = np.interp(
            elevation_angle,
            CHANNEL_PARAMETER["elevation_deg"],
            CHANNEL_PARAMETER["los_sigma_sf_db"]
        )
        shadow_fading_db = np.random.normal(0, sigma_sf)
        clutter_loss_db = 0.0
    else:
        sigma_sf = np.interp(
            elevation_angle,
            CHANNEL_PARAMETER["elevation_deg"],
            CHANNEL_PARAMETER["nlos_sigma_sf_db"]
        )
        shadow_fading_db = np.random.normal(0, sigma_sf)
        clutter_loss_db = np.interp(
            elevation_angle,
            CHANNEL_PARAMETER["elevation_deg"],
            CHANNEL_PARAMETER["nlos_cl_db"]
        )
    # --- System parameters from the table ---
    UE_TX_EIRP_DBM = 23.01
    SAT_RX_GAIN_DBI = 24.0
    FC_GHZ = 2.0
    BANDWIDTH_HZ = 0.4e6
    NOISE_FIGURE_DB = 5.0
    # --- Free-space path loss ---
    # distance_km in km, fc in GHz
    fspl_db = 92.45 + 20 * np.log10(FC_GHZ) + 20 * np.log10(distance_km)
    path_loss_db = fspl_db + shadow_fading_db + clutter_loss_db
    noise_dbm = -174 + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
    snr_db = UE_TX_EIRP_DBM + SAT_RX_GAIN_DBI - path_loss_db - noise_dbm
    #print(snr_db)
    return snr_db > 0

def channel_visibility(UE_location, satellite, min_elevation,t):
    difference = satellite - UE_location
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > min_elevation, alt.degrees, distance.km

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

def load_fixed_satellites(filename="fixed_satellite_pool.json"):
    with open(filename, "r", encoding="utf-8") as f:
        records = json.load(f)
    # 重新載入與 generate_satellite_pool.py 相同的 TLE
    satellites = load.tle_file(
        'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
        filename='starlink_tle.txt',
        reload=False
    )
    sat_dict = {
        sat.model.satnum: sat
        for sat in satellites
    }
    real_sats = []
    for rec in records:
        norad_id = rec["norad_id"]
        if norad_id not in sat_dict:
            raise ValueError(
                f"Satellite {norad_id} ({rec['name']}) not found in TLE file."
            )
        real_sats.append(sat_dict[norad_id])
    return real_sats

def load_ps_tables(filename="group_ps_table.npz"):
    data = np.load(
        filename,
        allow_pickle=True
    )
    group_weight_table = data["group_weight_table"]
    group_ps_table = data["group_ps_table"]
    print(f"Loaded {len(group_weight_table)} RAOs")
    print(f"Group weight table shape: {group_weight_table.shape}")
    print(f"Group ps table shape: {group_ps_table.shape}")
    return group_weight_table, group_ps_table

def main(RHO, SECONDS, NUM_UE,MODE, SEED, NUM_EPOCHS, IMBALANCE_EPSILON=0.01):
    # 模式設定
    np.random.seed(SEED) # 固定隨機種子以確保可重現性
    print(f"--- Simulation Start ---")
    print(f"Mode: {MODE}, Active rate: {RHO},  Time Slots: {SECONDS}")
    print(f"Satellite selection imbalance epsilon: {IMBALANCE_EPSILON}")
    
    # 設定觀察點 (台北)
    geo = wgs84.latlon(25.03, 121.56)
    ts = load.timescale()
    start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc)
    #t_start = ts.from_datetime(start_dt)
    real_sats = load_fixed_satellites(
        "fixed_satellite_pool.json"
    )
    #設定controller
    #載入其他預運算資料
    group_weight_table, group_ps_table = load_ps_tables()
    ctrl = controller(group_weight_table=group_weight_table, group_ps_table=group_ps_table)
    # 將真實衛星「封裝」進您的 Simulation Class
    sat_list = []
    for i, real_sat in enumerate(real_sats):
        s = satellite(id=0, skyfield_sat=real_sat) #這裡不賦予id，因為後面會篩選可見衛星池，id會對不上，所以直接在satellite class裡面用list index當id。
        sat_list.append(s)
    #print(f"Complete building environment: {len(sat_list)} satellites loaded.")
    
    trao = 100
    throughput_history = [] # 記錄每個 Slot 的成功數
    n_history = [] # 記錄每個 Slot 的 N_estimate
    
    # 主模擬迴圈
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數

    first_ps_table = next((table for table in group_ps_table if len(table) > 0), None)
    if first_ps_table is None:
        raise ValueError("group_ps_table has no groups; cannot infer satellite count.")
    table_sat_count = len(next(iter(first_ps_table.values())))
    if len(sat_list) < table_sat_count:
        raise ValueError(
            f"Fixed satellite pool has {len(sat_list)} satellites, but group_ps_table expects {table_sat_count}."
        )
    active_sat_pool = sat_list[:table_sat_count]

    print(f"Active Sat Pool Size: {len(active_sat_pool)}")

    for i in range(len(active_sat_pool)):
        sat = active_sat_pool[i]
        ctrl.add_satellite(sat) #Controller只加入active_sat_pool裡的衛星
        sat.assign_id(i) #為每個衛星分配新的ID
    ctrl.set_agent() #在加入衛星後初始化Agent，讓Agent知道目前的衛星列表和數量
    if MODE != 1:
        NUM_EPOCHS = 1 #如果不是RL模式，就只跑一個epoch，因為不需要訓練過程

    each_epo_plr = []
    each_epo_thr = []
    each_epo_reward = []
    expected_tables = Load_estimator.precompute_expected_tables(Z=sat_list[0].Z, Nmax=1000) #預計算期望值表，傳入Z值和Nmax上限
    visibility_recorder = {}
    for epoch in range(NUM_EPOCHS): #目前只跑一個epoch，之後可以增加多個epoch來觀察學習趨勢
        n_history = [] # 記錄每個 Slot 的 N_estimate
        if epoch == 0:
            ue_list = []
            R_km = 200.0  # 想要維持強烈幾何落差，建議設 200.0 ~ 300.0 km
            c = [25.03, 121.56] # 台北中心點

            # 2. 將公里半徑轉換為經緯度的最大邊界（軸向縮放）
            lat_bound = R_km / 111.0
            lon_bound = R_km / 100.0
            for i in range(NUM_UE):
                # 3. 圓形均勻抽樣
                r = np.sqrt(np.random.uniform(0, 1)) # sqrt 確保地表真均勻分佈（不會往中心擠）
                theta = np.random.uniform(0, 2 * np.pi)
                # 4. 映射回實際經緯度
                lat = c[0] + (r * np.sin(theta)) * lat_bound
                lon = c[1] + (r * np.cos(theta)) * lon_bound
                
                ue_list.append(UE(location=[lat, lon], id=i, rho=RHO))
                ctrl.ue_list = ue_list #將UE列表傳給controller，讓controller可以在需要的時候訪問UE資訊
            # 重置 controller 的暫存狀態
        ctrl.reset_agent()
        throughput_history = []
        for ue in ue_list:
            ue.acb_selection_count = 0
            ue.acb_policy_fallback_count = 0
        #start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc) #每個epoch重置時間，之後可能會改成從不同時間開始
        #重置衛星狀態
        for sat in active_sat_pool:
            sat.ue_pre = {}
            sat.N_i = sat.N_s = sat.N_c = 0
            sat.actual_lambda = 0

        for n in range(RAO_COUNTS): #統一用n，表示現在是在第幾個RAO
            # --- 更新時間與產生封包 ---
            arrival_mask = np.random.rand(NUM_UE) < (RHO * 1000 / trao)
            for i, ue in enumerate(ue_list):
                ue.new_time(bursty=arrival_mask[i])

            current_ms = n * trao
            current_dt = start_dt + timedelta(milliseconds=current_ms)
            current_t = ts.from_datetime(current_dt)
            # --- 衛星移動與可見衛星列表更新 ---
            if n % 5 == 0 or n == 1: #每5個RAO更新一次可見衛星列表，因為衛星移動不會太快
                visible_count = 0
                #sat_visibility_counts = np.zeros(len(active_sat_pool))
                if epoch == 0:
                    record_dict = {}
                    for ue in ue_list:
                        ue.acquire_visible_sat(active_sat_pool, current_t,MODE,ctrl.sat_num)
                        visible_count += len(ue.visible_satellites)
                        #for sat in ue.visible_satellites:
                            #sat_visibility_counts[sat.id] += 1
                        record_dict[ue.id] = ue.visible_satellites
                    visibility_recorder[n] = record_dict
                else:
                    for ue in ue_list:
                        ue.visible_satellites = visibility_recorder[n][ue.id]
                        visible_count += len(ue.visible_satellites)
                # 計算變異係數 (CV)，衡量衛星間潛在競爭壓力的不均勻性
                #cv_visibility = np.std(sat_visibility_counts) / np.mean(sat_visibility_counts)
                avg_visible = visible_count / NUM_UE
                if n % 50 == 0 and n>0:
                    print(f"RAO {n}: Average visible satellites per UE: {avg_visible:.2f}")
                    #print(f"RAO {n}: Coefficient of Variation for Satellite Visibility: {cv_visibility:.2f}")
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
            
            ctrl.actualPi = real_counts / NUM_UE #更新真實pi供測試參考
            #Controller-side processing
            ctrl.set_group_probabilities_for_rao(
                n,
                use_convex_solver=(MODE == 1),
                imbalance_epsilon=IMBALANCE_EPSILON,
            )
            # 中文註解：先用預計算表格與目前 group selection policy 算出預估 p_s，後面會和本輪實際觀察值比較。
            p_s = calculate_ps(ctrl,n,group_weight_table, group_ps_table)
            #print(f"Precomputed p_s for RAO {n}: {p_s:.4f}")
            if n == 0:
                Lambda = np.zeros(ctrl.sat_num)
                current_n_hat = ctrl.N_estimate
            else:
                Lambda = ctrl.load_estimator(expected_tables) #每個RAO都呼叫一次load estimator，並且傳入預計算好的期望值表
                ctrl.satellite_selection(Lambda=Lambda,MODE=MODE, n=n, target_location=geo, t=current_t, epoch=epoch)
                ctrl.backoff_control(total_load=sum(Lambda), rho=(RHO * 1000 / trao), p_d = ue_list[0].QoS_requirement, p_s=p_s, K=ctrl.sat_num, Z=sat_list[0].Z,MODE=MODE,n=n)
                current_n_hat = ctrl.N_estimate
            n_history.append(current_n_hat)
            
            if n % 50 == 0:
                print(f"Current N_tilde: {ctrl.N_estimate}, Total Load (Lambda): {sum(Lambda)}, Backoff rate: {ctrl.p_b}", end='\n')

            for ue in ue_list:
                if ue.active: #只對active的UE計算ACB和決定順序
                    ue.acquire_SIB(ctrl)

            # 中文註解：記錄本輪 UE 執行 RA 前的通道成功/失敗累積值，用來計算本輪真實 p_s。
            channel_success_before = sum(ue.transmission_success for ue in ue_list)
            channel_fail_before = sum(ue.transmission_fail for ue in ue_list)

            # UE-side processing
            for ue in ue_list:
                # 如果通過 ACB，會呼叫 sat.receive_preamble()
                if ue.active: 
                    ue.ACB_test()

            # 中文註解：真實 p_s 定義為本輪實際嘗試 RA 的 UE 中，通道判定成功的比例；若本輪無嘗試則不計算。
            channel_success_after = sum(ue.transmission_success for ue in ue_list)
            channel_fail_after = sum(ue.transmission_fail for ue in ue_list)
            slot_channel_success = channel_success_after - channel_success_before
            slot_channel_fail = channel_fail_after - channel_fail_before
            slot_channel_attempts = slot_channel_success + slot_channel_fail
            if slot_channel_attempts > 0 and n % 50 == 0:
                real_p_s = slot_channel_success / slot_channel_attempts
                print(f"RAO {n}: Real p_s={real_p_s:.4f}, Precomputed p_s={p_s:.4f}, Diff={real_p_s - p_s:+.4f}")
            #else:
                #print(f"RAO {n}: Real p_s=N/A (no RA attempts), Precomputed p_s={p_s:.4f}")
                
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
        # [更改 5] 週期結束後的統計與模型存檔
        # 統計結果
        total_success_packets = sum(throughput_history)
        total_lost_packets = sum(ue.loss for ue in ue_list)
        channel_failure_rates = sum(ue.transmission_fail for ue in ue_list) / (sum(ue.transmission_success for ue in ue_list) + sum(ue.transmission_fail for ue in ue_list))
        policy_fallback_count = sum(ue.acb_policy_fallback_count for ue in ue_list)
        acb_selection_count = sum(ue.acb_selection_count for ue in ue_list)
        policy_fallback_rate = policy_fallback_count / acb_selection_count if acb_selection_count > 0 else 0.0
        avg_throughput = total_success_packets / (RAO_COUNTS * trao / 1000)  # packets per second
        plr = 1 - total_success_packets/(total_success_packets+total_lost_packets)
        print(f"----------Episode {epoch+1} Complete.----------")
        print(f"Total Successful Accesses: {total_success_packets}")
        print(f"Total Dropped Packets: {total_lost_packets}")
        print(f"Average Throughput (packets/second): {avg_throughput:.2f}")
        print(f"Packet Loss Rate: {plr:.4f}")
        print(f"Channel Failure Rate: {channel_failure_rates:.4f}")
        print(f"ACB Policy Fallback Frequency: {policy_fallback_count}/{acb_selection_count} ({policy_fallback_rate:.4f})")

        each_epo_plr.append(plr)
        each_epo_thr.append(avg_throughput)
        each_epo_reward.append(np.mean(ctrl.history_reward))
    epo_history = {
        "throughput": each_epo_thr,
        "plr": each_epo_plr,
        "reward": each_epo_reward
    }
    return avg_throughput, plr, n_history, ctrl.actual, ctrl.observe_pi, ctrl.history_reward, epo_history #印出最後一個epoch的效能


    
