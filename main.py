import numpy as np
from skyfield.api import load, wgs84
from skyfield.framelib import itrs  # 此次 2026/6/9 凌晨 visibility 加速修改：批次版仰角計算需要衛星的 ITRS/ECEF 座標。
import orbit
from datetime import datetime, timezone, timedelta  # 必須有 timedelta
import Load_estimator, backoff_control, N_estimate, selection
import json
from scipy.special import erf
from scenario_time import get_tle_scenario_metadata, load_starlink_tles

class controller:
    def __init__(self, group_weight_table=None, group_ps_table=None):
        self.satellites = []
        self.sat_num = len(self.satellites) 
        self.Dmax = 5 #Delay budget的最大值
        self.p_b = np.zeros(self.Dmax)
        self.observe_pi = np.ones(self.Dmax) / self.Dmax
        self.success_state_ratio = np.ones(self.Dmax) / self.Dmax
        self.group_weight_table = group_weight_table
        self.group_ps_table = group_ps_table
        self.A_by_group = {}
        self.N_estimate = 0
        self.rls = N_estimate.RLSEstimator(initial_N=self.N_estimate)
        self.actualPi = np.zeros(self.Dmax + 1) #用來記錄每個pi的真實值，供測試參考，包含 idle state
        self.actual = []
        self.history_reward = []
        self.ue_list = []
        self.selection_solver_failures = 0
        self.last_load_indicator = None
        self.load_aware_eta = 1.0
        self.load_aware_load_ema = None
        self.load_aware_load_ema_history = []
        self.adaptive_epsilon_load_ema = 0.0
        self.adaptive_epsilon_history = []
        self.previous_A_by_group = None
        self.selection_policy_variation_history = []
    def set_group_probabilities_for_rao(self, n, selection_mode, use_convex_solver=False, imbalance_epsilon=0.01, preamble_count=None):
        if selection_mode == 5:
            self.A_by_group = {}
            return
        if self.group_weight_table is None:
            self.A_by_group = {}
            return
        weights = self.group_weight_table[n]
        ps_by_group = self.group_ps_table[n] if self.group_ps_table is not None else None
        if selection_mode == 4:
            # Mode 4 baseline: every group selects its highest-elevation satellite,
            # which is stored as the first satellite index in the ordered group key.
            current_probabilities = {}
            for group in weights.keys():
                group_key = tuple(group)
                if self.sat_num <= 0:
                    raise ValueError("Cannot build A_g before satellites are registered.")
                top_satellite = int(group_key[0])
                if top_satellite < 0 or top_satellite >= self.sat_num:
                    raise ValueError(
                        f"Mode 4 group {group_key} points to satellite {top_satellite}, "
                        f"but sat_num is {self.sat_num}."
                    )
                a_g = np.zeros(self.sat_num)
                a_g[top_satellite] = 1.0
                current_probabilities[group_key] = a_g
            self.A_by_group = current_probabilities
            return
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
                #print(
                 #   f"Warning: convex satellite selection failed at RAO {n}; "
                  #  f"falling back to uniform A_g. Reason: {exc}"
                #)

        current_probabilities = {}
        for group in weights.keys():
            group_key = tuple(group)
            if self.sat_num <= 0:
                raise ValueError("Cannot build A_g before satellites are registered.")
            current_probabilities[group_key] = np.ones(self.sat_num) / self.sat_num
        self.A_by_group = current_probabilities

    def record_selection_policy_variation(self, n, selection_mode):
        # Diagnostic only: measure how much the broadcast group selection
        # probabilities A_g change between consecutive RAOs.
        current_policy = {
            tuple(group): np.asarray(probabilities, dtype=float).copy()
            for group, probabilities in self.A_by_group.items()
        }

        if selection_mode in (3, 5):
            self.selection_policy_variation_history.append({
                "time_slot": n,
                "mean_tv": np.nan,
                "weighted_tv": np.nan,
                "max_tv": np.nan,
                "common_group_count": 0,
            })
            self.previous_A_by_group = current_policy
            return

        if self.previous_A_by_group is None:
            self.selection_policy_variation_history.append({
                "time_slot": n,
                "mean_tv": np.nan,
                "weighted_tv": np.nan,
                "max_tv": np.nan,
                "common_group_count": 0,
            })
            self.previous_A_by_group = current_policy
            return

        common_groups = set(current_policy).intersection(self.previous_A_by_group)
        if not common_groups:
            self.selection_policy_variation_history.append({
                "time_slot": n,
                "mean_tv": np.nan,
                "weighted_tv": np.nan,
                "max_tv": np.nan,
                "common_group_count": 0,
            })
            self.previous_A_by_group = current_policy
            return

        weights = self.group_weight_table[n] if self.group_weight_table is not None else {}
        tv_values = []
        weight_values = []
        for group in common_groups:
            current = current_policy[group]
            previous = self.previous_A_by_group[group]
            if len(current) != len(previous):
                continue
            tv = 0.5 * np.sum(np.abs(current - previous))
            tv_values.append(tv)
            weight_values.append(float(weights.get(group, 0.0)))

        if len(tv_values) == 0:
            mean_tv = np.nan
            weighted_tv = np.nan
            max_tv = np.nan
        else:
            tv_values = np.asarray(tv_values, dtype=float)
            weight_values = np.asarray(weight_values, dtype=float)
            mean_tv = float(np.mean(tv_values))
            max_tv = float(np.max(tv_values))
            if np.sum(weight_values) > 0:
                weighted_tv = float(np.average(tv_values, weights=weight_values))
            else:
                weighted_tv = mean_tv

        self.selection_policy_variation_history.append({
            "time_slot": n,
            "mean_tv": mean_tv,
            "weighted_tv": weighted_tv,
            "max_tv": max_tv,
            "common_group_count": len(tv_values),
        })
        self.previous_A_by_group = current_policy

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
    def backoff_control(self, total_load, rho, p_d, p_s, K, Z, backoff_mode,n):
        #實作backoff control
        if backoff_mode == 1:
            denominator = np.sum(self.observe_pi * (1 - self.p_b)) *p_s
            #denominator = np.sum(self.actualPi[1:] * (1 - self.p_b)) *p_s
            N_tilde = self.N_estimation(Lambda=total_load, denominator=denominator)
            self.N_estimate = N_tilde
        else:
            N_tilde = self.N_estimate
        self.p_b, self.observe_pi = backoff_control.backoff_control(N_tilde, self.p_b, rho, self.Dmax, p_d, p_s, K, Z, backoff_mode, total_load, self.success_state_ratio)
        if n % 10 == 0:
            #print(f"Actual Pi: {self.actualPi}, Observed Pi: {self.observe_pi}")
            self.actual.append(list(self.actualPi))
        #print(f"Backoff control updated: p_b={self.p_b}, pi={self.observe_pi}")
        return
    def update_success_state_ratio(self, success_states):
        counts = np.zeros(self.Dmax)
        for state in success_states:
            if state is not None and 1 <= state <= self.Dmax:
                counts[int(state) - 1] += 1
        total = np.sum(counts)
        if total > 0:
            self.success_state_ratio = counts / total
        return self.success_state_ratio
    def satellite_selection(self, Lambda,MODE,n,target_location,t):
        avg_load = np.mean(Lambda)
        reward = -np.mean((Lambda - avg_load) ** 2)
        self.history_reward.append(reward)
        if n % 50 == 0 and n>0:
            print(f"Current group selection policies: {len(self.A_by_group)} groups")
            print(f"Current Reward: {reward}")
        return
    def N_estimation(self, Lambda, denominator):
        current_N = self.rls.update(Lambda, denominator)
        return current_N
    def reset_agent(self):
        self.last_state = None
        self.last_action_idx = None

    def update_load_aware_load_indicator(self, load, beta):
        # Mode 5 uses the same EMA idea as the adaptive proposed method, but
        # applies it per satellite so the load penalty does not overreact.
        beta = np.clip(beta, 0.0, 1.0)
        load = np.maximum(np.asarray(load, dtype=float), 0.0)
        if len(load) != self.sat_num:
            raise ValueError(
                f"Mode 5 EMA load length {len(load)} does not match sat_num {self.sat_num}."
            )
        if self.load_aware_load_ema is None:
            self.load_aware_load_ema = np.zeros(self.sat_num, dtype=float)
        self.load_aware_load_ema = (
            (1.0 - beta) * self.load_aware_load_ema
            + beta * load
        )
        self.last_load_indicator = self.load_aware_load_ema.copy()
        self.load_aware_load_ema_history.append({
            "load_ema": self.load_aware_load_ema.copy(),
            "load_raw": load.copy(),
            "beta": float(beta),
        })
        return self.last_load_indicator

    def adaptive_imbalance_epsilon(self, total_load, total_preambles, epsilon_min, epsilon_max, alpha, beta):
        # Mode 6: adapt the convex load-balance tolerance from a smoothed
        # normalized load. Higher load gives a smaller, more conservative epsilon.
        if total_preambles <= 0:
            raise ValueError("total_preambles must be positive for adaptive epsilon.")
        beta = np.clip(beta, 0.0, 1.0)
        total_load = max(float(total_load), 0.0)
        self.adaptive_epsilon_load_ema = (
            (1.0 - beta) * self.adaptive_epsilon_load_ema
            + beta * total_load
        )
        normalized_load = self.adaptive_epsilon_load_ema / float(total_preambles)
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-alpha * normalized_load)
        epsilon = float(np.clip(epsilon, epsilon_min, epsilon_max))
        self.adaptive_epsilon_history.append({
            "load_ema": self.adaptive_epsilon_load_ema,
            "normalized_load": normalized_load,
            "epsilon": epsilon,
        })
        return epsilon
     
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
    def receive_preamble(self,ue_id,angle, distance, remaining_budget=None, fixed_channel_success_prob=None):
        # 模擬 UE隨機選取一個 Preamble (0 到 Z-1)
        if fixed_channel_success_prob is None:
            channel_success = channel_calculator(angle, distance)
        else:
            channel_success = np.random.rand() < fixed_channel_success_prob

        if channel_success:
            chosen_preamble = np.random.randint(0, self.Z)
            self.ue_pre[ue_id] = (chosen_preamble, remaining_budget)
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
            preamble = value[0]
            if preamble in seen_values:
                duplicates.add(preamble)
            else:
                seen_values.add(preamble)
        for ue, value in self.ue_pre.items():
            preamble, remaining_budget = value
            if preamble not in duplicates:
                success_list.append((ue, remaining_budget))
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
        self.current_delay_raos = 0
        self.success_delay_raos = []
        self.group = None #這個UE被分配到的衛星群組，初始為None，之後會根據可見衛星列表更新
        self.transmission_success = 0 #實際成功傳輸的次數，減掉self.success後就是碰撞次數
        self.transmission_fail = 0
        self.active = False #Boolean
        self.active_prob = rho
        self.QoS_requirement = [0.2,0.2,0.2,0.2,0.2] #對應不同delay budget的QoS需求，總和為1
        self.visible_satellites = []
        self.all_satellites = []
        self.A_g = None
        self.selection_mode = None
        self.fixed_channel_success_prob = None
        self.channel_success_prob = np.zeros(0)
        self.load_indicator = None
        self.load_aware_eta = 1.0
        self.acb_selection_count = 0
        self.acb_policy_fallback_count = 0
        self.geo = wgs84.latlon(self.location[0], self.location[1])
        # 此次 2026/6/9 凌晨 visibility 加速修改：UE 位置固定，先保存 ECEF 與 ENU 單位向量，避免每個 RAO 重複建立地面座標。
        self.lat_rad = np.deg2rad(self.location[0])
        self.lon_rad = np.deg2rad(self.location[1])
        self.ecef_km = self.geo.itrs_xyz.km
        self.enu_east = np.array([-np.sin(self.lon_rad), np.cos(self.lon_rad), 0.0])
        self.enu_north = np.array([
            -np.sin(self.lat_rad) * np.cos(self.lon_rad),
            -np.sin(self.lat_rad) * np.sin(self.lon_rad),
            np.cos(self.lat_rad),
        ])
        self.enu_up = np.array([
            np.cos(self.lat_rad) * np.cos(self.lon_rad),
            np.cos(self.lat_rad) * np.sin(self.lon_rad),
            np.sin(self.lat_rad),
        ])
    def acquire_visible_sat(self,sat_list,current_time_obj,mode,num):
        self.visible_satellites = []
        self.all_satellites = list(sat_list)
        self.angle = np.zeros(len(sat_list))        
        self.distance = np.zeros(len(sat_list))
        if mode == 2:
            self.fixed_channel_success_prob = 1.0
            for sat in sat_list:
                self.visible_satellites.append(sat) #全部都看的到
        else:
            self.fixed_channel_success_prob = None
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
            self.current_delay_raos += 1
            if self.delay >= self.budget:
                self.active = False
                self.loss += 1
                self.delay = 0
                self.current_delay_raos = 0
        else:
            if bursty:
                self.new_packet()
    def new_packet(self):
        self.active = True
        r = np.random.rand()
        self.budget = np.random.choice([1, 2, 3, 4, 5], p=self.QoS_requirement) #根據QoS需求隨機分配delay budget
        self.delay = 0
        self.current_delay_raos = 1
        self.target_satellite = None
    def acquire_SIB(self,ctrl):
        #取得系統資訊，包含backoff機率和衛星選擇資訊
        #print(f"UE {self.id}, group {self.group}")
        self.p_b = ctrl.p_b
        if self.selection_mode == 5:
            self.load_indicator = ctrl.last_load_indicator
            self.load_aware_eta = ctrl.load_aware_eta
            self.A_g = None
            return
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
        remaining_budget = self.budget - self.delay
        if remaining_budget <= 0:
            self.active = False
            self.loss += 1
            self.delay = 0
            self.current_delay_raos = 0
            return

        if r < self.p_b[remaining_budget - 1]:
            backoff = True
        if self.selection_mode in (3, 5):
            # VU and Mode 5 select only from the UE-side visible set.
            candidate_satellites = self.visible_satellites
        else:
            candidate_satellites = self.all_satellites if len(self.all_satellites) > 0 else self.visible_satellites
        if not backoff and len(candidate_satellites) > 0:
            self.acb_selection_count += 1
            if self.selection_mode == 3:
                target_sat = np.random.choice(candidate_satellites)
                self.execute_RA(target_sat)
                return
            if self.selection_mode == 5:
                load_indicator = self.load_indicator
                candidate_ids = [sat.id for sat in candidate_satellites]
                if load_indicator is None or len(load_indicator) <= max(candidate_ids):
                    self.acb_policy_fallback_count += 1
                    target_sat = max(candidate_satellites, key=lambda sat: self.angle[sat.id])
                    self.execute_RA(target_sat)
                    return
                if len(self.channel_success_prob) <= max(candidate_ids):
                    self.acb_policy_fallback_count += 1
                    target_sat = max(candidate_satellites, key=lambda sat: self.angle[sat.id])
                    self.execute_RA(target_sat)
                    return
                load_scale = float(candidate_satellites[0].Z)
                link_probabilities = np.asarray(self.channel_success_prob)[candidate_ids]
                probabilities = link_probabilities * np.exp(
                    -self.load_aware_eta
                    * np.maximum(np.asarray(load_indicator)[candidate_ids], 0.0)
                    / load_scale
                )
                prob_sum = np.sum(probabilities)
                if prob_sum <= 0 or not np.isfinite(prob_sum):
                    self.acb_policy_fallback_count += 1
                    target_sat = max(candidate_satellites, key=lambda sat: self.angle[sat.id])
                    self.execute_RA(target_sat)
                    return
                chosen_idx = np.random.choice(len(candidate_satellites), p=probabilities / prob_sum)
                target_sat = candidate_satellites[chosen_idx]
                self.execute_RA(target_sat)
                return
            if self.fixed_channel_success_prob is not None:
                target_sat = np.random.choice(candidate_satellites)
                self.execute_RA(target_sat)
                return
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
        r = target_sat.receive_preamble(
            self.id,
            self.angle[target_sat.id],
            self.distance[target_sat.id],
            remaining_budget=self.budget - self.delay,
            fixed_channel_success_prob=self.fixed_channel_success_prob,
        )
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
            self.success_delay_raos.append(self.current_delay_raos)
            self.current_delay_raos = 0
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

def _normal_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def estimate_channel_success_probability(elevation_angle, distance_km):
    elevation_angle = np.asarray(elevation_angle, dtype=float)
    distance_km = np.asarray(distance_km, dtype=float)
    valid = (elevation_angle > 0) & (distance_km > 0)
    elevation_angle = np.clip(elevation_angle, 0, 90)

    p_los = np.interp(
        elevation_angle,
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [0.0, 0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998],
    )
    los_sigma = np.interp(
        elevation_angle,
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [1.79, 1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72],
    )
    nlos_sigma = np.interp(
        elevation_angle,
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [8.93, 8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52],
    )
    nlos_clutter_loss = np.interp(
        elevation_angle,
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [20.87, 19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30],
    )

    fspl_db = 92.45 + 20 * np.log10(2.0) + 20 * np.log10(np.maximum(distance_km, 1e-12))
    noise_dbm = -174 + 10 * np.log10(0.4e6) + 5.0
    base_margin_db = 23.01 + 24.0 - fspl_db - noise_dbm
    p_success = (
        p_los * _normal_cdf(base_margin_db / los_sigma)
        + (1.0 - p_los) * _normal_cdf((base_margin_db - nlos_clutter_loss) / nlos_sigma)
    )
    return np.where(valid, p_success, 0.0)

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

def update_visibility_batch(ue_list, sat_list, current_time_obj, mode, min_elevation=0, chunk_size=5000):
    # 此次 2026/6/9 凌晨 visibility 加速修改：每個 RAO 仍完整更新 visibility，但改成批次 ECEF/ENU 投影，避免 UE*衛星 次 Skyfield altaz 呼叫。
    sat_count = len(sat_list)
    if sat_count == 0:
        for ue in ue_list:
            ue.visible_satellites = []
            ue.all_satellites = []
            ue.angle = np.zeros(0)
            ue.distance = np.zeros(0)
            ue.group = None
            ue.selection_mode = mode
            ue.fixed_channel_success_prob = None
            ue.channel_success_prob = np.zeros(0)
        return 0

    sat_snapshot = list(sat_list)
    if mode == 2:
        # 此次 2026/6/9 凌晨 visibility 加速修改：保留 ideal mode 的原始語意，所有 UE 都視為可見全部衛星。
        for ue in ue_list:
            ue.visible_satellites = list(sat_snapshot)
            ue.all_satellites = list(sat_snapshot)
            ue.angle = np.zeros(sat_count)
            ue.distance = np.zeros(sat_count)
            ue.group = None
            ue.selection_mode = mode
            ue.fixed_channel_success_prob = 1.0
            ue.channel_success_prob = np.ones(sat_count)
        return len(ue_list) * sat_count

    # VU and Mode 5 use the same 10-degree UE-side visibility filter.
    visibility_min_elevation = 10 if mode in (3, 5) else min_elevation

    for sat in sat_snapshot:
        if sat.id < 0 or sat.id >= sat_count:
            raise ValueError(
                f"Satellite id {sat.id} is outside angle/distance array length {sat_count}."
            )

    # 此次 2026/6/9 凌晨 visibility 加速修改：衛星位置只和當前 RAO 時間有關，每顆衛星在本 RAO 只轉一次 ITRS/ECEF。
    sat_ecef_km = np.stack(
        [sat.skyfield_sat.at(current_time_obj).frame_xyz(itrs).km for sat in sat_snapshot],
        axis=0,
    )

    visible_count = 0
    for start in range(0, len(ue_list), chunk_size):
        # 此次 2026/6/9 凌晨 visibility 加速修改：分批處理 UE，維持矩陣化速度，同時避免大量 UE 時一次配置過大的 delta 矩陣。
        chunk = ue_list[start:start + chunk_size]
        ue_ecef_km = np.vstack([ue.ecef_km for ue in chunk])
        east_vectors = np.vstack([ue.enu_east for ue in chunk])
        north_vectors = np.vstack([ue.enu_north for ue in chunk])
        up_vectors = np.vstack([ue.enu_up for ue in chunk])

        delta = sat_ecef_km[None, :, :] - ue_ecef_km[:, None, :]
        up_component = np.einsum("nkd,nd->nk", delta, up_vectors)
        east_component = np.einsum("nkd,nd->nk", delta, east_vectors)
        north_component = np.einsum("nkd,nd->nk", delta, north_vectors)
        horizontal_distance = np.hypot(east_component, north_component)
        elevation_deg = np.degrees(np.arctan2(up_component, horizontal_distance))
        distance_km = np.linalg.norm(delta, axis=2)
        channel_success_prob = estimate_channel_success_probability(elevation_deg, distance_km) if mode == 5 else None
        visible_mask = elevation_deg > visibility_min_elevation
        sorted_indices = np.argsort(elevation_deg, axis=1)[:, ::-1]

        for local_idx, ue in enumerate(chunk):
            # 此次 2026/6/9 凌晨 visibility 加速修改：回填既有 UE 欄位，讓後續 ACB/RA 邏輯沿用原本資料介面。
            ue.all_satellites = list(sat_snapshot)
            ue.angle = elevation_deg[local_idx].copy()
            ue.distance = distance_km[local_idx].copy()
            ue.selection_mode = mode
            ue.fixed_channel_success_prob = None
            ue.channel_success_prob = channel_success_prob[local_idx].copy() if mode == 5 else np.zeros(sat_count)
            ue.load_indicator = None
            visible_indices = np.flatnonzero(visible_mask[local_idx])
            ue.visible_satellites = [sat_snapshot[i] for i in visible_indices]
            visible_count += len(visible_indices)
            if mode == 5:
                ue.group = None
            elif sat_count >= 2:
                ue.group = (
                    sat_snapshot[sorted_indices[local_idx, 0]].id,
                    sat_snapshot[sorted_indices[local_idx, 1]].id,
                )
            else:
                ue.group = None

    return visible_count

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
    satellites = load_starlink_tles()
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

def _npz_scalar(data, key):
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value

def load_ps_tables(filename="group_ps_table.npz", scenario_metadata=None, expected_sat_norad_ids=None):
    data = np.load(
        filename,
        allow_pickle=True
    )
    group_weight_table = data["group_weight_table"]
    group_ps_table = data["group_ps_table"]
    # Only selection mode 3 consumes this table; keep other modes compatible
    # with older preselection files that may not contain the baseline column.
    mode3_visible_random_ps_table = (
        data["mode3_visible_random_ps_table"]
        if "mode3_visible_random_ps_table" in data.files
        else None
    )
    if scenario_metadata is not None:
        required_keys = ["scenario_start_dt_iso", "tle_file_sha256"]
        for key in required_keys:
            if key not in data.files:
                raise ValueError(
                    f"{filename} does not contain {key}. Regenerate it with satellite_preselection.py."
                )
        table_start_dt = str(_npz_scalar(data, "scenario_start_dt_iso"))
        table_tle_hash = str(_npz_scalar(data, "tle_file_sha256"))
        if table_start_dt != scenario_metadata["start_dt_iso"]:
            raise ValueError(
                f"{filename} start time mismatch. Table={table_start_dt}, "
                f"TLE-derived scenario={scenario_metadata['start_dt_iso']}."
            )
        if table_tle_hash != scenario_metadata["tle_file_sha256"]:
            raise ValueError(
                f"{filename} was generated from a different TLE file. Regenerate it."
            )
    if expected_sat_norad_ids is not None:
        if "sat_norad_ids" not in data.files:
            raise ValueError(f"{filename} does not contain sat_norad_ids. Regenerate it.")
        table_sat_ids = np.asarray(data["sat_norad_ids"], dtype=int)
        expected_sat_ids = np.asarray(expected_sat_norad_ids, dtype=int)
        if not np.array_equal(table_sat_ids, expected_sat_ids):
            raise ValueError(
                f"{filename} satellite IDs do not match fixed_satellite_pool.json. Regenerate it."
            )
    print(f"Loaded {len(group_weight_table)} RAOs")
    print(f"Group weight table shape: {group_weight_table.shape}")
    print(f"Group ps table shape: {group_ps_table.shape}")
    if mode3_visible_random_ps_table is not None:
        print(f"VU ps table shape: {mode3_visible_random_ps_table.shape}")
    return group_weight_table, group_ps_table, mode3_visible_random_ps_table

def main(RHO, SECONDS, NUM_UE, MODE, SEED, IMBALANCE_EPSILON=0.01, USE_REAL_PS=False, LOAD_AWARE_ETA=5.0, LOAD_AWARE_LOAD_EMA_BETA=0.2, ADAPTIVE_EPSILON_MIN=1e-4, ADAPTIVE_EPSILON_MAX=1e-2, ADAPTIVE_EPSILON_ALPHA=2.0, ADAPTIVE_EPSILON_BETA=0.2, QOS_DISTRIBUTION=None):
    # 模式設定
    np.random.seed(SEED) # 固定隨機種子以確保可重現性
    if MODE == 0:
        selection_mode = 0
        backoff_mode = 0
    else:
        selection_mode = MODE[0]
        backoff_mode = MODE[1]
    print(f"--- Simulation Start ---")
    print(f"Mode: {MODE}, Arrival rate lambda: {RHO} packets/s,  Time Slots: {SECONDS}")
    if selection_mode == 0:
        print("Mode 0: visibility heterogeneity test")
    else:
        print(f"Selection mode: {selection_mode}, Backoff mode: {backoff_mode}")
    if selection_mode == 6:
        print("Satellite selection imbalance epsilon: adaptive")
    else:
        print(f"Satellite selection imbalance epsilon: {IMBALANCE_EPSILON}")
    print(f"Use lagged real p_s: {USE_REAL_PS}")
    print(f"Load-aware eta: {LOAD_AWARE_ETA}")
    if selection_mode == 5:
        print(f"Mode 5 load EMA beta: {LOAD_AWARE_LOAD_EMA_BETA}")
    # Optional QoS sweep hook: keep the legacy uniform distribution when no
    # experiment-specific delay budget distribution is provided.
    if QOS_DISTRIBUTION is None:
        qos_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
    else:
        qos_distribution = np.asarray(QOS_DISTRIBUTION, dtype=float)
        if qos_distribution.shape != (5,):
            raise ValueError("QOS_DISTRIBUTION must contain five probabilities for delay budgets 1..5.")
        if np.any(qos_distribution < 0) or not np.all(np.isfinite(qos_distribution)):
            raise ValueError("QOS_DISTRIBUTION must contain finite non-negative probabilities.")
        qos_sum = float(np.sum(qos_distribution))
        if qos_sum <= 0:
            raise ValueError("QOS_DISTRIBUTION must have a positive sum.")
        qos_distribution = qos_distribution / qos_sum
    print(f"Delay budget distribution: {qos_distribution.tolist()}")
    if selection_mode == 6:
        print(
            "Adaptive epsilon: "
            f"min={ADAPTIVE_EPSILON_MIN}, max={ADAPTIVE_EPSILON_MAX}, "
            f"alpha={ADAPTIVE_EPSILON_ALPHA}, beta={ADAPTIVE_EPSILON_BETA}"
        )
    
    # 設定觀察點 (台北)
    geo = wgs84.latlon(25.03, 121.56)
    ts = load.timescale()
    scenario_metadata = get_tle_scenario_metadata()
    start_dt = scenario_metadata["start_dt"]
    print(f"Scenario start time from TLE median epoch: {scenario_metadata['start_dt_iso']}")
    #t_start = ts.from_datetime(start_dt)
    real_sats = load_fixed_satellites()
    #設定controller
    #載入其他預運算資料
    expected_sat_norad_ids = [int(sat.model.satnum) for sat in real_sats]
    group_weight_table, group_ps_table, mode3_visible_random_ps_table = load_ps_tables(
        scenario_metadata=scenario_metadata,
        expected_sat_norad_ids=expected_sat_norad_ids,
    )
    if selection_mode == 5:
        group_weight_table = None
        group_ps_table = None
    ctrl = controller(group_weight_table=group_weight_table, group_ps_table=group_ps_table)
    ctrl.load_aware_eta = LOAD_AWARE_ETA
    # 將真實衛星「封裝」進您的 Simulation Class
    sat_list = []
    for i, real_sat in enumerate(real_sats):
        s = satellite(id=0, skyfield_sat=real_sat) #這裡不賦予id，因為後面會篩選可見衛星池，id會對不上，所以直接在satellite class裡面用list index當id。
        sat_list.append(s)
    #print(f"Complete building environment: {len(sat_list)} satellites loaded.")
    
    trao = 100
    rho_rao = 1 - np.exp(-RHO * trao / 1000)
    print(f"RAO-level arrival probability: {rho_rao:.6f}")
    throughput_history = [] # 記錄每個 Slot 的成功數
    n_history = [] # 記錄每個 Slot 的 N_estimate
    
    # 主模擬迴圈
    RAO_COUNTS = SECONDS * 1000 // trao  # 將秒數轉換成640ms的Slot數

    if selection_mode == 5:
        active_sat_pool = sat_list
    else:
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
    expected_tables = Load_estimator.precompute_expected_tables(Z=sat_list[0].Z, Nmax=1000) #預計算期望值表，傳入Z值和Nmax上限
    n_history = [] # 記錄每個 Slot 的 N_estimate
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
        
        ue = UE(location=[lat, lon], id=i, rho=RHO)
        ue.QoS_requirement = qos_distribution.copy()
        ue_list.append(ue)
    ctrl.ue_list = ue_list #將UE列表傳給controller，讓controller可以在需要的時候訪問UE資訊

    ctrl.reset_agent()
    throughput_history = []
    last_real_p_s = None
    ps_history = []
    p_b_history = []
    for ue in ue_list:
        ue.acb_selection_count = 0
        ue.acb_policy_fallback_count = 0
    #重置衛星狀態
    for sat in active_sat_pool:
        sat.ue_pre = {}
        sat.N_i = sat.N_s = sat.N_c = 0
        sat.actual_lambda = 0

    for n in range(RAO_COUNTS): #統一用n，表示現在是在第幾個RAO
        # --- 更新時間與產生封包 ---
        arrival_mask = np.random.rand(NUM_UE) < rho_rao
        for i, ue in enumerate(ue_list):
            ue.new_time(bursty=arrival_mask[i])

        current_ms = n * trao
        current_dt = start_dt + timedelta(milliseconds=current_ms)
        current_t = ts.from_datetime(current_dt)
        # --- 衛星移動與可見衛星列表更新 ---
        visible_count = update_visibility_batch(ue_list, active_sat_pool, current_t, selection_mode)
        avg_visible = visible_count / NUM_UE
        if n % 50 == 0 and n>0:
            print(f"RAO {n}: Average visible satellites per UE: {avg_visible:.2f}")
        if avg_visible < 1:
            print("Warning: Too few visible satellites on average. The simulation scenario is not feasible. Ending simulation.")
            return
        if selection_mode == 0: #測試模式，不是真的跑模擬
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
        
        ctrl.actualPi = np.concatenate(([idle_ue_count / NUM_UE], real_counts / NUM_UE)) #更新真實pi供測試參考，index 0 為 idle state
        if n == 0:
            Lambda = np.zeros(ctrl.sat_num)
            current_n_hat = ctrl.N_estimate
        else:
            Lambda = ctrl.load_estimator(expected_tables) #每個RAO都呼叫一次load estimator，並且傳入預計算好的期望值表
            current_n_hat = ctrl.N_estimate
        # Mode 5 smooths the latest available load report before making the
        # current RAO's load-and-link-aware satellite selection decision.
        if selection_mode == 5:
            ctrl.update_load_aware_load_indicator(Lambda, LOAD_AWARE_LOAD_EMA_BETA)
        else:
            ctrl.last_load_indicator = Lambda.copy()
        effective_imbalance_epsilon = IMBALANCE_EPSILON
        if selection_mode == 6:
            # Mode 6 keeps the proposed convex selection, but tightens epsilon
            # when the EMA-smoothed normalized load becomes high.
            effective_imbalance_epsilon = ctrl.adaptive_imbalance_epsilon(
                total_load=sum(Lambda),
                total_preambles=ctrl.sat_num * sat_list[0].Z,
                epsilon_min=ADAPTIVE_EPSILON_MIN,
                epsilon_max=ADAPTIVE_EPSILON_MAX,
                alpha=ADAPTIVE_EPSILON_ALPHA,
                beta=ADAPTIVE_EPSILON_BETA,
            )
            if n % 50 == 0:
                print(f"Adaptive epsilon at RAO {n}: {effective_imbalance_epsilon:.6f}")
        #Controller-side processing
        ctrl.set_group_probabilities_for_rao(
            n,
            selection_mode=selection_mode,
            use_convex_solver=(selection_mode in (1, 6)),
            imbalance_epsilon=effective_imbalance_epsilon,
            preamble_count=sat_list[0].Z,
        )
        ctrl.record_selection_policy_variation(n, selection_mode)
        # Compute the precomputed p_s from the group selection policy; optionally replace it with lagged real p_s for control.
        if selection_mode == 2:
            precomputed_p_s = 1.0
        elif selection_mode in (3, 5):
            # Mode 3 uses the preselection table for uniform random selection
            # over satellites visible above 10 degrees, matching its UE-side rule.
            if mode3_visible_random_ps_table is None:
                raise ValueError(
                    "Mode 3/5 requires mode3_visible_random_ps_table. "
                    "Regenerate group_ps_table.npz with satellite_preselection.py."
            )
            precomputed_p_s = mode3_visible_random_ps_table[n]
        else:
            precomputed_p_s = calculate_ps(ctrl,n,group_weight_table, group_ps_table)
        p_s = last_real_p_s if (USE_REAL_PS and last_real_p_s is not None) else precomputed_p_s
        #print(f"Precomputed p_s for RAO {n}: {p_s:.4f}")
        if n > 0:
            ctrl.satellite_selection(Lambda=Lambda,MODE=selection_mode, n=n, target_location=geo, t=current_t)
            ctrl.backoff_control(total_load=sum(Lambda), rho=rho_rao, p_d = ue_list[0].QoS_requirement, p_s=p_s, K=ctrl.sat_num, Z=sat_list[0].Z,backoff_mode=backoff_mode,n=n)
            p_b_history.append(ctrl.p_b.copy())
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
        if slot_channel_attempts > 0:
            real_p_s = slot_channel_success / slot_channel_attempts
            last_real_p_s = real_p_s
            ps_history.append({
                "time_slot": n,
                "real": real_p_s,
                "precomputed": precomputed_p_s,
                "control": p_s,
                "error": real_p_s - precomputed_p_s,
            })
            if n % 50 == 0:
                print(
                    f"RAO {n}: Real p_s={real_p_s:.4f}, "
                    f"Precomputed p_s={precomputed_p_s:.4f}, "
                    f"Control p_s={p_s:.4f}, "
                    f"Diff={real_p_s - precomputed_p_s:+.4f}"
                )
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
        success_states_in_this_slot = []
        for sat in sat_list:
            # 回傳該衛星成功接收的 UE ID 列表
            successes = sat.check_RA_success()
            for ue_id, remaining_budget in successes:
                total_success_ids_in_this_slot.append(ue_id)
                success_states_in_this_slot.append(remaining_budget)

        # 記錄本時間點的總吞吐量
        throughput_history.append(len(total_success_ids_in_this_slot))

        # --- 回傳結果給 UE (更新狀態) ---
        for ue in ue_list:
            if ue.active: #只有active的UE才會收到反饋，並且可能改變狀態
                ue.receive_feedback(total_success_ids_in_this_slot)
        ctrl.update_success_state_ratio(success_states_in_this_slot)
    # 統計結果
    total_success_packets = sum(throughput_history)
    total_lost_packets = sum(ue.loss for ue in ue_list)
    success_delay_raos = [
        delay_raos
        for ue in ue_list
        for delay_raos in ue.success_delay_raos
    ]
    avg_delay_ms = np.mean(success_delay_raos) * trao if len(success_delay_raos) > 0 else np.nan
    channel_failure_rates = sum(ue.transmission_fail for ue in ue_list) / (sum(ue.transmission_success for ue in ue_list) + sum(ue.transmission_fail for ue in ue_list))
    policy_fallback_count = sum(ue.acb_policy_fallback_count for ue in ue_list)
    acb_selection_count = sum(ue.acb_selection_count for ue in ue_list)
    policy_fallback_rate = policy_fallback_count / acb_selection_count if acb_selection_count > 0 else 0.0
    policy_variation_values = np.array(
        [item["weighted_tv"] for item in ctrl.selection_policy_variation_history],
        dtype=float,
    )
    finite_policy_variation = policy_variation_values[np.isfinite(policy_variation_values)]
    selection_policy_variation_mean = float(np.mean(finite_policy_variation)) if len(finite_policy_variation) > 0 else np.nan
    selection_policy_variation_max = float(np.max(finite_policy_variation)) if len(finite_policy_variation) > 0 else np.nan
    avg_throughput = total_success_packets / (RAO_COUNTS * trao / 1000)  # packets per second
    plr = 1 - total_success_packets/(total_success_packets+total_lost_packets)
    print(f"----------Simulation Complete.----------")
    print(f"Total Successful Accesses: {total_success_packets}")
    print(f"Total Dropped Packets: {total_lost_packets}")
    print(f"Average Throughput (packets/second): {avg_throughput:.2f}")
    print(f"Packet Loss Rate: {plr:.4f}")
    print(f"AverageDelay (ms): {avg_delay_ms:.2f}" if np.isfinite(avg_delay_ms) else "AverageDelay (ms): N/A")
    print(f"Channel Failure Rate: {channel_failure_rates:.4f}")
    print(f"ACB Policy Fallback Frequency: {policy_fallback_count}/{acb_selection_count} ({policy_fallback_rate:.4f})")
    if np.isfinite(selection_policy_variation_mean):
        print(
            f"A_g Policy Variation: mean={selection_policy_variation_mean:.4f}, "
            f"max={selection_policy_variation_max:.4f}"
        )
    else:
        print("A_g Policy Variation: N/A")

    run_history = {
        "throughput": avg_throughput,
        "plr": plr,
        "AverageDelay": avg_delay_ms,
        "average_delay_ms": avg_delay_ms,
        "average_delay_raos": np.mean(success_delay_raos) if len(success_delay_raos) > 0 else np.nan,
        "reward": np.mean(ctrl.history_reward),
        "ps_history": ps_history,
        "p_b_history": p_b_history,
        "adaptive_epsilon_history": ctrl.adaptive_epsilon_history,
        "load_aware_load_ema_history": ctrl.load_aware_load_ema_history,
        "selection_policy_variation_history": ctrl.selection_policy_variation_history,
        "selection_policy_variation_mean": selection_policy_variation_mean,
        "selection_policy_variation_max": selection_policy_variation_max,
    }
    return avg_throughput, plr, n_history, ctrl.actual, ctrl.observe_pi, ctrl.history_reward, run_history


    
