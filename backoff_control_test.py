import matplotlib.pyplot as plt
import numpy as np
import old as main

# --- 1. 設定測試參數 ---
num = 10000
# 定義多個不同的 rho 進行掃描 (例如從 0.002 到 0.02，共 6 個點)
rho_list = np.array([0.002, 0.005, 0.01, 0.015, 0.02])

# 定義要比較的兩種情境 (Mode 3 與 Mode 4)
# 格式為: {情境名稱: (Mode, 軌道面或衛星配置參數)}
scenarios = {
    'Baseline: Non-state dependent Backoff': (3, 3),
    'Proposed Backoff Control': (4, 3)
}
#'Baseline: Non-state dependent Backoff': 'o', 
 #          'Proposed Backoff Control': 's'}

# 用於儲存最終繪圖數據的結構
# 結構如: {'Scenario Name': [plr_at_rho1, plr_at_rho2, ...]}
plr_results = {name: [] for name in scenarios.keys()}

print("Start simulation scanning over different rho values...")

# --- 2. 執行模擬運行與數據收集 ---
for rho in rho_list:
    for name, (mode, orbit_param) in scenarios.items():
        print(f"Running {name} with rho = {rho}...")
        
        # 執行 main.main 並取得回傳值
        # 注意：此處維持原設計的參數順序，將目前的 rho 傳入
        a, b, c, d, e, f, g = main.main(
            rho,          # 傳入目前的 rho
            orbit_param,  # 軌道面數量/可見衛星數量
            100, 
            num, 
            mode,         # 傳入目前的 mode (3 或 4)
            42, 
            50
        )
        
        # 收集該 rho 底下的最終封包遺失率 (PLR)
        plr_results[name].append(b)

print("--- Simulation Complete ---")

# --- 3. 繪製不同 rho 下的 PLR 折線圖 ---
plt.figure(figsize=(10, 6))

# 為不同情境設定固定顏色與標記，方便辨識
markers = {'Baseline: Non-state dependent Backoff': 'o', 
           'Proposed Backoff Control': 's'}
colors = {'Baseline: Non-state dependent Backoff': 'tab:blue', 
          'Proposed Backoff Control': 'tab:orange'}

for name in scenarios.keys():
    plt.plot(
        rho_list, 
        plr_results[name], 
        marker=markers[name], 
        color=colors[name], 
        linewidth=2, 
        markersize=6, 
        label=name
    )

plt.xlabel('Arrival Rate (rho)', fontsize=11)
plt.ylabel('Packet Loss Rate (PLR)', fontsize=11)
plt.title('PLR Comparison', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
# 修正重點：將圖例放在圖表外面右側
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
# 優化 X 軸顯示，確保每個測試的 rho 都有刻度
plt.xticks(rho_list)

plt.tight_layout()
plt.show()

# --- 4. 終端機數據打印對照 ---
print("\n--- Summary Table (PLR) ---")
print(f"{'rho':<10}{'Baseline PLR':<20}{'Proposed PLR':<20}")
print("-" * 50)
for idx, rho in enumerate(rho_list):
    m3_plr = plr_results['Baseline: Non-state dependent Backoff'][idx]
    m4_plr = plr_results['Proposed Backoff Control'][idx]
    print(f"{rho:<10.4f}{m3_plr:<20.4f}{m4_plr:<20.4f}")