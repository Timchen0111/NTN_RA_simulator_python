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
num = 10000
a,b,c,d,e=main.main(0.05, 1, 500, num, 1, 1)
plt.figure(figsize=(10, 6))
plt.plot(range(len(c)), c, label='Estimated N (N_tilde)', color='blue', linewidth=1.5)
plt.axhline(y=num, color='red', linestyle='--', label=f'True N ({num})') # 畫出真實值參考線


plt.title('Convergence of Population Estimation (N_tilde)')
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# 將 list 轉換為 numpy array 以便進行切片操作

data_matrix = np.array(d)
plt.figure(figsize=(10, 6))

# 定義顏色與標籤，方便辨識 State 0 ~ 4
colors = ['blue', 'green', 'red', 'orange', 'purple']
labels = [f'State {i}' for i in range(5)]

for i in range(5):
    plt.plot(range(len(data_matrix)), data_matrix[:, i], 
             label=labels[i], color=colors[i], linewidth=1.5)
    plt.axhline(y=e[i], color=colors[i], linestyle='--', label=f'Observe Pi ({e[i]})') # 畫出真實值參考線

plt.title('Convergence of State Distributions ($\pi_n$)', fontsize=14)
plt.xlabel('Time Slot (n)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()