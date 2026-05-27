import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
import orbit
from datetime import datetime, timezone, timedelta  # 必須有 timedelta
import Load_estimator, backoff_control, N_estimate, selection
import json
from concurrent.futures import ThreadPoolExecutor

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
        self.agent = selection.SatelliteSelectionAgent(satellite_list=self.satellites, S_max=2, mem_length=5)
        self.last_state = None
        self.last_action_idx = None
        self.S = np.ones(len(self.satellites))
    def add_satellite(self, satellite):
        self.satellites.append(satellite)
        self.sat_num = len(self.satellites) 


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
    def calculate_elevation_angle(self,target_location,t):
        difference = self.skyfield_sat - target_location
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        return alt.degrees
    
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
        for sat in sat_list:
            if is_visible(self.geo, sat.skyfield_sat, min_elevation=20, t=current_time_obj): #所以這行是Bottleneck，當UE數量大就會變很慢。
                self.visible_satellites.append(sat.id)
        

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
        set1 = set(u1.visible_satellites)
        set2 = set(u2.visible_satellites)
        
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
            sat_appearance[sat] = sat_appearance.get(sat, 0) + 1
    
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

def main(NUM_SAT, SECONDS, NUM_UE):
    # 取得真實衛星列表 (Top-N 軌道面)
    # 設定模擬開始時間
    ts = load.timescale()
    start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc)
    t_start = ts.from_datetime(start_dt)
    geo = wgs84.latlon(25.03, 121.56)
    
    real_sats = orbit.get_relevant_rail_planes(t_start, geo, top_n=NUM_SAT)
    sat_list = []
    for i, real_sat in enumerate(real_sats):
        s = satellite(id=0, skyfield_sat=real_sat) #這裡不賦予id，因為後面會篩選可見衛星池，id會對不上，所以直接在satellite class裡面用list index當id。
        sat_list.append(s)
    
    trao = 100
    # 原主模擬迴圈：這裡改成讓所有UE跑一次，但只紀錄可見衛星並存入CSV檔，不涉及後續的RA流程。
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數
    active_sat_pool = []
    ctrl = controller()

    for sat in sat_list:
    #暫時用區域中心點的衛星仰角篩選一部份的衛星，允許的仰角閾值更加小因為只是初步篩選。
        mid_dt = start_dt + timedelta(seconds=SECONDS/2)
        if is_visible(geo, sat.skyfield_sat, min_elevation=10, t=ts.from_datetime(mid_dt)): 
            active_sat_pool.append(sat) #這些就是模擬中我們系統考慮的衛星池，後續不再更動。
    print(f"Active Sat Pool Size: {len(active_sat_pool)}")
    for i in range(len(active_sat_pool)):
        sat = active_sat_pool[i]
        ctrl.add_satellite(sat) #Controller只加入active_sat_pool裡的衛星
        sat.assign_id(i) #為每個衛星分配新的ID
    ctrl.set_agent() #在加入衛星後初始化Agent，讓Agent知道目前的衛星列表和數量
    ue_list = []

    for i in range(NUM_UE):
        c = [25.03, 121.56] #centers[np.random.choice([0, 1])]
        raw_lat_offset = np.random.uniform(-2, 2)
        raw_lon_offset = np.random.uniform(-2, 2)
        # 立方變換：(offset^3) / 2.25 確保範圍大致維持在原有尺度，但極度集中
        lat = c[0] + (raw_lat_offset ** 3) / 2.25
        lon = c[1] + (raw_lon_offset ** 3) / 2.25
        ue_list.append(UE(location=[lat, lon], id=i, rho=0)) #RHO不重要
    
    start_dt = datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc) #每個epoch重置時間，之後可能會改成從不同時間開始
    env_config = {
        "satellites": [
            {"id": sat.id, "name": sat.skyfield_sat.name} for sat in active_sat_pool
        ],
        "ues": [
            {"id": ue.id, "location": ue.location} for ue in ue_list
        ],
        "rao_records": [] # 存放動態的可見性數據
    }

    for n in range(RAO_COUNTS):

        current_ms = n * trao
        current_dt = start_dt + timedelta(milliseconds=current_ms)
        current_t = ts.from_datetime(current_dt)

        record = {
            "RAO": n,
            "timestamp": current_dt.isoformat(),
            "UE_data": {}
        }

        if n % 5 == 0 or n == 0:
            # --- 這裡是最強優化點：預先計算所有衛星的位置 ---
            # 這樣每個時間點每顆衛星只會呼叫一次 .at(t)
            sat_positions = [sat.skyfield_sat.at(current_t) for sat in active_sat_pool]
            print(f"RAO {n}: Parallel updating visibility...")
            
            with ThreadPoolExecutor() as executor:
                # 傳入預算好的 sat_positions
                results = list(executor.map(
                    lambda ue: update_ue_optimized(ue, sat_positions, active_sat_pool, current_t), 
                    ue_list
                ))
            visible_count = 0
            print(f"RAO {n}: Updating visible satellite lists for all UEs...")
            # 寫回紀錄
            for ue_id, vis_sats in results:
                record["UE_data"][ue_id] = vis_sats
                visible_count += len(vis_sats)
            avg_visible = visible_count / NUM_UE
            if n % 50 == 0 and n > 0:
                print(f"RAO {n}: Average visible satellites per UE: {avg_visible:.2f}")
            env_config["rao_records"].append(record)

    with open("visible_recorder.json", "w") as f:
        json.dump(env_config, f, indent=2)

def update_ue_optimized(ue_data, sat_positions, active_sat_pool, current_t):
    visible_ids = []
    # 1. 建立地理位置物件
    ue_location = wgs84.latlon(ue_data.location[0], ue_data.location[1])
    # 2. 關鍵修正：必須呼叫 .at(current_t) 
    # 這樣 ue_at_t 才會變成具備 xyz 坐標的「特定時間位置物件」
    ue_at_t = ue_location.at(current_t)
    for i, sat_pos in enumerate(sat_positions):
        # 現在 sat_pos (Geocentric) 與 ue_at_t 之間可以執行減法了
        difference = sat_pos - ue_at_t
        # 計算仰角
        # 因為已經是相對向量，直接 altaz() 即可
        alt, az, distance = difference.altaz()
        if alt.degrees > 20: 
            visible_ids.append(active_sat_pool[i].id)
    return ue_data.id, visible_ids

def is_visible(UE_location, satellite, min_elevation,t):
    difference = satellite - UE_location
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > min_elevation

def update_ue(ue,active_sat_pool,current_t):
    ue.acquire_visible_sat(active_sat_pool, current_t)
    return ue.id, ue.visible_satellites
#main(2,20,100)
main(4, 150, 40000)
