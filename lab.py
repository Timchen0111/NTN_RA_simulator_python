import numpy as np
import Load_estimator as l
import main
import matplotlib.pyplot as plt
'''
# 引用先前定義的函數
# precompute_expected_tables, load_estimator

def test_mom_performance():
    # --- 1. 設定測試參數 ---
    Z = 54           # 前導碼總數
    Nmax = 1000       # 估計搜尋上限
    test_lambdas = np.arange(1, 401, 10)  # 測試負載從 1 到 400，每隔 10 點取樣
    num_trials = 50   # 每個負載點重複實驗次數，以取得平均表現
    
    # 預計算理論表
    tables = l.precompute_expected_tables(Z, Nmax)
    
    ground_truth = []
    estimations = []

    print(f"Start to test MoM Estimator (Z={Z}, Nmax={Nmax})...")

    # --- 2. 執行模擬測試 ---
    for true_lambda in test_lambdas:
        trial_estimates = []
        for _ in range(num_trials):
            # 模擬隨機接入過程：每個 UE 隨機選一個 preamble
            # 產生 true_lambda 個隨機整數，範圍 [0, Z-1]
            selections = np.random.randint(0, Z, size=true_lambda)
            
            # 計算觀測值 N_i, N_s, N_c
            counts = np.bincount(selections, minlength=Z)
            n_i = np.sum(counts == 0)
            n_s = np.sum(counts == 1)
            n_c = np.sum(counts > 1)
            
            # 執行估計 (包裝成 array 格式以符合函數輸入)
            hat_lambda = l.load_estimator(np.array([n_i]), np.array([n_s]), np.array([n_c]), tables)
            trial_estimates.append(hat_lambda[0])
            
        ground_truth.append(true_lambda)
        estimations.append(np.mean(trial_estimates))

    # --- 3. 計算誤差指標 ---
    ground_truth = np.array(ground_truth)
    estimations = np.array(estimations)
    mae = np.mean(np.abs(ground_truth - estimations))
    
    print(f"Complete! MAE: {mae:.2f}")

    # --- 4. 繪製結果圖表 ---
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth, ground_truth, 'r--', label='Ideal (Ground Truth)')
    plt.scatter(ground_truth, estimations, color='blue', label='MoM Estimation')
    plt.xlabel('True Load (Number of UEs)')
    plt.ylabel('Estimated Load (Lambda Hat)')
    plt.title('MoM Estimator Performance Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

# 執行測試
if __name__ == "__main__":
    test_mom_performance()
}

'''
'''
# 1. 設定實驗參數
rho = [0.2,0.4,0.6] 
SEEDS = [13] 
MODES = [1, 2] 

# 儲存最終結果的字典
# 結構為 {mode: [(avg_tp, avg_sr), ...]}
final_results = {mode: [] for mode in MODES}

for count in rho:
    # 儲存當前 UE 數量下，各個 Mode 的加總數據
    mode_sums = {mode: [0.0, 0.0] for mode in MODES} # [tp_sum, sr_sum]
    
    for s in SEEDS:
        print(f"Running simulations for UE count: {count} with seed: {s}...")
        
        for mode in MODES:
            # 每次執行前重置種子，確保相同 UE 分佈與流量
            np.random.seed(s)
            
            # 呼叫 main 並取得回傳值 (throughput, success_rate)
            tp, sr = main.main(count, 1, 100, mode, s)
            
            mode_sums[mode][0] += tp
            mode_sums[mode][1] += sr
            
    # 計算平均值並存入最終結果
    num_seeds = len(SEEDS)
    for mode in MODES:
        avg_tp = mode_sums[mode][0] / num_seeds
        avg_sr = mode_sums[mode][1] / num_seeds
        final_results[mode].append((avg_tp, avg_sr))

# 2. 準備繪圖數據
labels = {1: 'with BACKOFF', 2: 'no BACKOFF'}
colors = {1: 'purple', 2: 'blue'}
markers = {1: 'o', 2: 's'}

# 3. 設定畫布
plt.figure(figsize=(14, 6))

# --- 左圖：Average Throughput 比較 ---
plt.subplot(1, 2, 1)
for mode in MODES:
    y_tp = [r[0] for r in final_results[mode]]
    plt.plot(rho, y_tp, marker=markers[mode], color=colors[mode], label=labels[mode])

plt.title('Active Rate vs Average Throughput')
plt.xlabel('Active Rate (RHO)')
plt.ylabel('Average Throughput (packets/second)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# --- 右圖：Success Rate 比較 ---
plt.subplot(1, 2, 2)
for mode in MODES:
    y_sr = [r[1] for r in final_results[mode]]
    plt.plot(rho, y_sr, marker=markers[mode], color=colors[mode], label=labels[mode])

plt.title('Active Rate vs Success Rate')
plt.xlabel('Active Rate (RHO)')
plt.ylabel('Success Rate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 4. 顯示結果
plt.tight_layout()
plt.show()
'''

#MODE說明：1.backoff control + satellite selection 2.只有backoff control 3.兩者都沒有
'''
import matplotlib.pyplot as plt
import numpy as np

# --- 模擬運行與數據收集 ---
num = 10000
modes = [1, 2, 3, 4]
results = {}
true_pi = None

for m in modes:
    # 執行 main.main 並取得回傳值
    # a: 最終負載, b: 成功率, c: N_tilde 歷史, d: Pi 歷史, e: 真實 Pi, f: reward 歷史
    a, b, c, d, e, f = main.main(0.05, 1, 100, num, m, 15)
    results[m] = {'N_tilde': c, 'Pi': d, 'Loads': a, 'SuccessRate': b, 'Reward': f}
    if m == 1: true_pi = e  # 紀錄基準真實值

# --- 圖表 1：人口估計收斂比較 (N_tilde) ---
plt.figure(figsize=(10, 6))
plt.axhline(y=num, color='black', linestyle='--', label=f'True N ({num})', alpha=0.6)

# 定義不同模式的樣式
configs = {
    1: {'label': 'MODE 1: RL + Backoff', 'color': 'blue'},
    2: {'label': 'MODE 2: Backoff Only', 'color': 'green'},
    3: {'label': 'MODE 3: No Control', 'color': 'red'},
    4: {'label': 'MODE 4: SimpleHeuristic + Backoff', 'color': 'orange'}
}

for m in modes:
    plt.plot(range(len(results[m]['N_tilde'])), results[m]['N_tilde'], 
             label=configs[m]['label'], color=configs[m]['color'], linewidth=1.5)

plt.title('Population Estimation Convergence (N_tilde) - Comparison')
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- 圖表 2：狀態分佈比較 (以特定 State 為例，例如 State 0) ---
# 若要畫出所有模式的 Pi 比較，通常建議選取最關鍵的 State (如 Idle 或 Collision)
plt.figure(figsize=(10, 6))
target_state = 0 # 假設我們觀察 State 0 的收斂情況
plt.axhline(y=true_pi[target_state], color='black', linestyle='--', 
            label=f'True Pi_{target_state}', alpha=0.6)

for m in modes:
    data_matrix = np.array(results[m]['Pi'])
    plt.plot(range(len(data_matrix)), data_matrix[:, target_state], 
             label=configs[m]['label'], color=configs[m]['color'], linewidth=1.5)

plt.title(f'State {target_state} Distribution Convergence Comparison', fontsize=14)
plt.xlabel('Time Slot (n)')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 假設 a: 成功次數, b: 碰撞率 (模擬結束後計算的平均值)
modes_list = ['MODE 1', 'MODE 2', 'MODE 3', 'MODE 4']
success_rates = [results[m]['SuccessRate'] for m in [1, 2, 3, 4]]

fig, ax1 = plt.subplots(figsize=(10, 6))

# 畫出成功率 (長條圖)
ax1.bar(modes_list, success_rates, color='skyblue', alpha=0.7, label='SuccessRate')
ax1.set_ylabel('Access Success Rate', color='blue', fontsize=12)

plt.title('Access Efficiency  Comparison', fontsize=14)
ax1.legend(loc='upper left')
plt.show()

# --- 圖表 2：Reward (負載平衡度) 變化比較 ---
plt.figure(figsize=(10, 6))

# 標註：Reward 越高 (越接近 0) 代表負載越平均
for m in modes:
    # 假設 results[m]['rewards'] 存放了每個時隙計算出的 -variance
    # 如果數據是 list，直接繪圖
    plt.plot(range(len(results[m]['Reward'])), results[m]['Reward'], 
             label=configs[m]['label'], color=configs[m]['color'], linewidth=1.2, alpha=0.8)

plt.title('Reward Convergence (Negative Load Variance) - Comparison')
plt.xlabel('Time Slot (n)')
plt.ylabel('Reward (Higher is Better)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

# --- 模擬運行與數據收集 (僅針對 MODE 1) ---
num = 10000
m = 1 # 僅跑 MODE 1: RL + Backoff
results = {}

# a: 最終負載, b: 成功率, c: N_tilde 歷史, d: Pi 歷史, e: 真實 Pi, f: reward 歷史
# 建議將 RAO 週期從 640ms 調小 (例如 100ms) 以增加 Episode 內的採樣點
a, b, c, d, e, f = main.main(0.01, 1, 100, num, m, 42)
results[m] = {
    'N_tilde': c, 
    'Pi': d, 
    'Loads': a, 
    'SuccessRate': b, 
    'Reward': f, 
    'TruePi': e
}

# --- 圖表 1：人口估計收斂 (N_tilde) ---
plt.figure(figsize=(10, 6))
plt.axhline(y=num, color='black', linestyle='--', label=f'True N ({num})', alpha=0.6)
plt.plot(range(len(results[m]['N_tilde'])), results[m]['N_tilde'], 
         label='MODE 1: RL + Backoff', color='blue', linewidth=1.5)

plt.title('Population Estimation Convergence (N_tilde) - MODE 1 Only')
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- 圖表 2：Reward 變化圖 (觀察收斂與 Visibility 失效) ---
plt.figure(figsize=(10, 6))
plt.plot(range(len(results[m]['Reward'])), results[m]['Reward'], 
         label='Reward (Negative Variance)', color='blue', linewidth=1.2)

# 標註：若 Reward 突然跳回 0，通常代表衛星已飛離範圍 (Lambda 歸零)
plt.title('Reward Variation over Time Slots')
plt.xlabel('Time Slot (n)')
plt.ylabel('Reward (-Var)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- 圖表 3：狀態分佈收斂 (以 State 0 為例) ---
plt.figure(figsize=(10, 6))
target_state = 0
true_pi_val = results[m]['TruePi'][target_state]
plt.axhline(y=true_pi_val, color='black', linestyle='--', label=f'True Pi_{target_state}', alpha=0.6)

data_matrix = np.array(results[m]['Pi'])
plt.plot(range(len(data_matrix)), data_matrix[:, target_state], 
         label=f'Observed Pi_{target_state}', color='blue', linewidth=1.5)

plt.title(f'State {target_state} Distribution Convergence')
plt.xlabel('Time Slot (n)')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"--- MODE 1 Test Complete ---")
print(f"Final Success Rate: {results[m]['SuccessRate']:.4f}")
