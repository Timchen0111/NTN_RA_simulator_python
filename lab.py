import numpy as np
import Load_estimator as l
import old as main
import matplotlib.pyplot as plt
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

'''
MODE說明
MODE1: RL + backoff (Proposed scheme)
MODE2: backoff only (With same satellite selection score)
MODE3: Baseline: No Satellite Selection, Non-state dependent backoff
MODE4: Heuristic satellite selection + backoff
MODE5: Iterative RL training and MODE4
MODE6: backoff + IDEAL SCENARIO, same visibility for all UEs, same satellite selection score
MODE7: Non-state dependent backoff + IDEAL SCENARIO, same visibility for all UEs, same satellite selection score
'''
'''
import matplotlib.pyplot as plt
import numpy as np

# --- 模擬運行與數據收集 ---
num = 20000
parameter_set = [[3,3,0.01],[4,3,0.01],[6,19,0.01]]  #Mode,軌道面數量/可見衛星數量
results = {}
true_pi = None

for i in range(len(parameter_set)):
    # 執行 main.main 並取得回傳值 
    # 注意：這裡的引數必須與你的 main 函式定義嚴格對齊
    # 傳入目前的軌道面數量 i，固定模式 m (建議先固定為舊版或特定模式進行變因隔離)
    a, b, c, d, e, f, g = main.main(parameter_set[i][2], parameter_set[i][1],30, num, parameter_set[i][0], 42, 50)
    results[i] = {
        'N_tilde': c, 
        'Pi': d, 
        'Loads': a, 
        'PLR': b, 
        'True_Pi': e,  # 將每一組實驗各自的 True Pi 存下來，因為衛星數變了，理論 True Pi 也會變
        'Reward': f
    }

# 為不同的軌道面數量動態產生顏色對比 (使用 colormap 讓圖表更專業)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(parameter_set)))
orbit_configs = {i: {'color': colors[idx]} for idx, i in enumerate(range(len(parameter_set)))}


# --- 圖表 1：不同軌道面數量下的人口估計收斂比較 (N_tilde) ---
plt.figure(figsize=(10, 6))
# 真實的總用戶數依然是固定的基準線
plt.axhline(y=num, color='black', linestyle='--', label=f'True N ({num})', alpha=0.6)

# 將集合 {} 改為列表 []，確保索引 i 能正確對應
#labels = ['Real Satellite Scenario, rho = 0.01', 'Real Satellite Scenario, rho = 0.005', 'Ideal Case: Uniform Visibility, rho = 0.01', 'Ideal Case: Uniform Visibility, rho = 0.005']
labels = ['Non-state dependent Backoff (Real Satellite)', 'Real Satellite Scenario', 'Ideal Case: Uniform Visibility']
for i in range(len(parameter_set)):
    plt.plot(range(len(results[i]['N_tilde'])), results[i]['N_tilde'], 
             label=labels[i], color=orbit_configs[i]['color'], linewidth=1.5)
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --- 圖表 2：不同軌道面數量下的 Pi 估計絕對誤差比較 (以 State 0 為例) ---
plt.figure(figsize=(10, 6))
target_state = 0  # 觀察 State 0 (Idle)

for i in range(len(parameter_set)):
    data_matrix = np.array(results[i]['Pi'])         # 模擬中估計的 Observed Pi 歷史紀錄
    current_true_pi = results[i]['True_Pi']          # 該配置下的系統真實 True Pi 固定值
    
    # 計算每個 Time Slot 的絕對誤差：|Observed_Pi - True_Pi|
    estimation_error = np.abs(data_matrix[:, target_state] - current_true_pi[target_state])
    
    # 修正重點：改用列表，並移除重複的 color 與 linewidth 參數
    plt.plot(range(len(estimation_error)), estimation_error, 
             label=labels[i], color=orbit_configs[i]['color'], linewidth=1.5)

# 理想狀況下的誤差基準線（誤差為 0）
plt.axhline(y=0, color='black', linestyle='--', alpha=0.6, label='Ideal Estimation (Zero Error)')

plt.title(f'State {target_state} Estimation Absolute Error across Different Orbit Scales', fontsize=12)
plt.xlabel('Time Slot (n)')
plt.ylabel('Absolute Error |Observed - True|')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# 填入你的實測數據
plr_real = results[0]["PLR"]
plr_real2 = results[1]["PLR"]
plr_ideal = results[2]["PLR"]
#plr_ideal2 = results[3]["PLR"]

plr_values = [plr_real, plr_real2, plr_ideal] #, plr_ideal2]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, plr_values, width=0.4)

# 在長條圖上方標註數值
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
    )

plt.ylabel("Packet Loss Rate (PLR)")
plt.title("PLR Comparison: Real vs. Ideal Case")
plt.ylim(0, max(plr_values) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"--- Test Complete ---")
print(f"Final PLR:")
print(f"Real satellite scenario, Non-state dependent Backoff: {results[0]['PLR']:.4f}")
print(f"Real satellite scenario: {results[1]['PLR']:.4f}")
print(f"Ideal case: {results[2]['PLR']:.4f}")
print(f"")
'''

import matplotlib.pyplot as plt
import numpy as np
import main

# --- 模擬運行與數據收集 (僅針對 MODE 1) ---
num = 10000
m = 1
results = {}

# a: 最終負載, b: 成功率, c: N_tilde 歷史, d: Pi 歷史, e: 真實 Pi, f: reward 歷史 g: episode history (包含 plr, reward, throughput)
#def main(RHO, NUM_SAT, SECONDS, NUM_UE,MODE, SEED, NUM_EPOCHS, IMBALANCE_EPSILON=1000)
a, b, c, d, e, f, g = main.main(0.01, 10, num, m, 42, 1,0.01)
results[m] = {
    'N_tilde': c, 
    'Pi': d, 
    'Loads': a, 
    'SuccessRate': b, 
    'Reward': f, 
    'TruePi': e,
    'EpisodeHistory': g
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

print(f"--- Test Complete ---")
print(f"Packet Loss Rate: {results[m]['SuccessRate']:.4f}")

epo_history = results[m]['EpisodeHistory']
# 設定繪圖風格
plt.style.use('seaborn-v0_8-muted')
epochs = np.arange(1, len(epo_history["plr"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Training Performance Analysis (Total Epochs: {len(epochs)})", fontsize=16)

# --- 圖一：Packet Loss Rate (PLR) 趨勢 ---
axes[0].plot(epochs, epo_history["plr"], color='#3498db', alpha=0.4, label='Raw PLR')
# 計算移動平均以觀察趨勢
if len(epochs) >= 10:
    ma_plr = np.convolve(epo_history["plr"], np.ones(10)/10, mode='valid')
    axes[0].plot(epochs[9:], ma_plr, color='#2980b9', linewidth=2, label='Moving Avg (10)')

# 如果有 MODE 4 的基準線數據，可以手動填入
#mode4_benchmark = 0.245  # 範例數值
#axes[0].axhline(y=mode4_benchmark, color='#e74c3c', linestyle='--', label='MODE 4 Baseline')

axes[0].set_title("Packet Loss Rate")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("PLR")
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# --- 圖二：Reward (Negative Load Variance) 收斂 ---
# Reward 反映了 RL 是否成功降低了衛星間的負載不均
axes[1].plot(epochs, epo_history["reward"], color='#f39c12')
axes[1].set_title("Learning Progress (Reward)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Average Reward")
axes[1].grid(True, linestyle=':', alpha=0.6)

# --- 圖三：System Throughput 吞吐量 ---
# 觀察在不同 UE 分佈下，總體吞吐量是否維持穩定
axes[2].plot(epochs, epo_history["throughput"], color='#27ae60')
axes[2].set_title("Total System Throughput")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Packets / Second")
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 最後在終端機輸出最終效能總結
final_10_plr = np.mean(epo_history["plr"][-10:])
print(f"\n" + "="*30)
print(f"Final Performance (Last 10 Epochs Avg):")
print(f"- Final PLR: {final_10_plr:.4f}")
print(f"- Throughput : {np.mean(epo_history['throughput'][-10:]):.2f} pkts/s")
print(f"="*30)

'''
# --- Epsilon sweep for convex group satellite selection ---
import matplotlib.pyplot as plt
import numpy as np
import main as simulator_main

EPSILON_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
EPSILON_RHO = 0.005
EPSILON_SECONDS = 30
EPSILON_NUM_UE = 10000
EPSILON_MODE = 1
EPSILON_SEED = 42
EPSILON_EPOCHS = 1

epsilon_results = []

for eps in EPSILON_VALUES:
    print(f"\nRunning epsilon sweep: epsilon={eps}")
    avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, epo_history = simulator_main.main(
        EPSILON_RHO,
        EPSILON_SECONDS,
        EPSILON_NUM_UE,
        EPSILON_MODE,
        EPSILON_SEED,
        EPSILON_EPOCHS,
        IMBALANCE_EPSILON=eps,
    )
    epsilon_results.append({
        "epsilon": eps,
        "plr": plr,
        "throughput": avg_throughput,
        "reward": np.mean(reward_history) if len(reward_history) > 0 else np.nan,
    })

epsilon_labels = [str(item["epsilon"]) for item in epsilon_results]
epsilon_plr = [item["plr"] for item in epsilon_results]
epsilon_throughput = [item["throughput"] for item in epsilon_results]
epsilon_reward = [item["reward"] for item in epsilon_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Convex Selection Epsilon Sweep", fontsize=16)

axes[0].plot(epsilon_labels, epsilon_plr, marker="o", color="#3498db")
axes[0].set_title("Packet Loss Rate")
axes[0].set_xlabel("Imbalance epsilon")
axes[0].set_ylabel("PLR")
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].plot(epsilon_labels, epsilon_throughput, marker="o", color="#27ae60")
axes[1].set_title("Average Throughput")
axes[1].set_xlabel("Imbalance epsilon")
axes[1].set_ylabel("Packets / Second")
axes[1].grid(True, linestyle=":", alpha=0.6)

axes[2].plot(epsilon_labels, epsilon_reward, marker="o", color="#f39c12")
axes[2].set_title("Average Reward")
axes[2].set_xlabel("Imbalance epsilon")
axes[2].set_ylabel("Reward")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n--- Epsilon Sweep Complete ---")
for item in epsilon_results:
    print(
        f"epsilon={item['epsilon']}: "
        f"PLR={item['plr']:.4f}, "
        f"throughput={item['throughput']:.2f}, "
        f"reward={item['reward']:.4f}"
    )
'''