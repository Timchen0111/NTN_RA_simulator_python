import main
import matplotlib.pyplot as plt
import numpy as np  # 記得 import numpy 來設定亂數種子

# 1. 執行模擬並收集數據
ue_counts = [50, 100, 150]
SEEDS = [42, 43, 44, 45, 46] # 跑 5 次不同的隨機流量取平均
#ue_counts = [50] # 測試用，正式跑的時候改回上面那行
results_proposed = []
results_traditional = []
results_fixed_acb = []

for count in ue_counts:
    # 用來暫存 5 次 Seed 的加總結果
    tp_prop_sum, sr_prop_sum = 0, 0
    tp_fixed_acb_sum, sr_fixed_acb_sum = 0, 0
    tp_trad_sum, sr_trad_sum = 0, 0
    print(f"Running simulations for UE count: {count} with seeds: {SEEDS}")
    for s in SEEDS:
        # 【關鍵】：每次呼叫 main 之前重置亂數種子，確保 3 種策略面對完全相同的 UE 分布與流量
        np.random.seed(s)
        r_prop = main.main(count, 1, 100, 2, s)
        
        np.random.seed(s)
        r_fixed_acb = main.main(count, 1, 100, 1, s)
        
        np.random.seed(s)
        r_trad = main.main(count, 1, 100, 0, s)
        
        # 累加結果
        tp_prop_sum += r_prop[0]
        sr_prop_sum += r_prop[1]
        
        tp_fixed_acb_sum += r_fixed_acb[0]
        sr_fixed_acb_sum += r_fixed_acb[1]
        
        tp_trad_sum += r_trad[0]
        sr_trad_sum += r_trad[1]
            
    # 取平均並存入陣列
    num_seeds = len(SEEDS)
    results_proposed.append((tp_prop_sum / num_seeds, sr_prop_sum / num_seeds))
    results_fixed_acb.append((tp_fixed_acb_sum / num_seeds, sr_fixed_acb_sum / num_seeds))
    results_traditional.append((tp_trad_sum / num_seeds, sr_trad_sum / num_seeds))

# 2. 準備繪圖數據
y_tp_prop = [r[0] for r in results_proposed]
y_sr_prop = [r[1] for r in results_proposed]

y_tp_trad = [r[0] for r in results_traditional]
y_sr_trad = [r[1] for r in results_traditional]

y_tp_fixed_acb = [r[0] for r in results_fixed_acb]
y_sr_fixed_acb = [r[1] for r in results_fixed_acb]

# 3. 設定畫布
plt.figure(figsize=(14, 6))

# --- 左圖：Average Throughput 比較 ---
plt.subplot(1, 2, 1)
plt.plot(ue_counts, y_tp_prop, marker='s', markersize=8, color='blue', label='SBC,DACB')
plt.plot(ue_counts, y_tp_trad, marker='d', markersize=8, color='red', linestyle='--', label='TRA,FACB')
plt.plot(ue_counts, y_tp_fixed_acb, marker='o', markersize=8, color='purple', linestyle='-.', label='SBC,FACB')
plt.title('UE Quantity vs Average Throughput')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Average Throughput (packets/slot)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# --- 右圖：Success Rate 比較 ---
plt.subplot(1, 2, 2)
plt.plot(ue_counts, y_sr_prop, marker='s', markersize=8, color='green', label='SBC,DACB')
plt.plot(ue_counts, y_sr_trad, marker='d', markersize=8, color='orange', linestyle='--', label='TRA,FACB')
plt.plot(ue_counts, y_sr_fixed_acb, marker='o', markersize=8, color='brown', linestyle='-.', label='SBC,FACB')
plt.title('UE Quantity vs Success Rate')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Success Rate')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# 4. 優化佈局並顯示
plt.tight_layout()
plt.show()