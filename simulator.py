import main
import matplotlib.pyplot as plt
import numpy as np  # 記得 import numpy 來設定亂數種子

# 1. 執行模擬並收集數據
ue_counts = [50, 75, 100, 125, 150]
#SEEDS = [42]
SEEDS = [13,18,26,34,39,51] # 跑 3 次不同的隨機流量取平均
#ue_counts = [50] # 測試用，正式跑的時候改回上面那行
results_sbc_d = []
results_sbc_f = []
results_tr_d = []
results_tr_f = []

for count in ue_counts:
    # 用來暫存 5 次 Seed 的加總結果
    tp_sbc_d_sum, sr_sbc_d_sum = 0, 0
    tp_sbc_f_sum, sr_sbc_f_sum = 0, 0
    tp_tr_d_sum, sr_tr_d_sum = 0, 0
    tp_tr_f_sum, sr_tr_f_sum = 0, 0
    
    for s in SEEDS:
        # 【關鍵】：每次呼叫 main 之前重置亂數種子，確保 3 種策略面對完全相同的 UE 分布與流量
        print(f"Running simulations for UE count: {count} with seeds: {s}...")
        np.random.seed(s)
        r_tr_d = main.main(count, 1, 100, 3, s)

        np.random.seed(s)
        r_sbc_d = main.main(count, 1, 100, 2, s)
        
        np.random.seed(s)
        r_sbc_f = main.main(count, 1, 100, 1, s)
        
        np.random.seed(s)
        r_tr_f = main.main(count, 1, 100, 0, s)
        
        # 累加結果
        tp_sbc_d_sum += r_sbc_d[0]
        sr_sbc_d_sum += r_sbc_d[1]
        
        tp_sbc_f_sum += r_sbc_f[0]
        sr_sbc_f_sum += r_sbc_f[1]
        
        tp_tr_d_sum += r_tr_d[0]
        sr_tr_d_sum += r_tr_d[1]

        tp_tr_f_sum += r_tr_f[0]
        sr_tr_f_sum += r_tr_f[1]
            
    # 取平均並存入陣列
    num_seeds = len(SEEDS)
    results_sbc_d.append((tp_sbc_d_sum / num_seeds, sr_sbc_d_sum / num_seeds))
    results_sbc_f.append((tp_sbc_f_sum / num_seeds, sr_sbc_f_sum / num_seeds))
    results_tr_d.append((tp_tr_d_sum / num_seeds, sr_tr_d_sum / num_seeds))
    results_tr_f.append((tp_tr_f_sum / num_seeds, sr_tr_f_sum / num_seeds))

# 2. 準備繪圖數據
y_tp_sbc_d = [r[0] for r in results_sbc_d]
y_sr_sbc_d = [r[1] for r in results_sbc_d]

y_tp_sbc_f = [r[0] for r in results_sbc_f]
y_sr_sbc_f = [r[1] for r in results_sbc_f]

y_tp_tr_d = [r[0] for r in results_tr_d]
y_sr_tr_d = [r[1] for r in results_tr_d]

y_tp_tr_f = [r[0] for r in results_tr_f]
y_sr_tr_f = [r[1] for r in results_tr_f]

# 3. 設定畫布
plt.figure(figsize=(14, 6))

# --- 左圖：Average Throughput 比較 ---
plt.subplot(1, 2, 1)
plt.plot(ue_counts, y_tp_sbc_d, marker='s', markersize=8, color='blue', label='SBC,DACB')
plt.plot(ue_counts, y_tp_tr_d, marker='d', markersize=8, color='red', linestyle='--', label='TRA,DACB')
plt.plot(ue_counts, y_tp_sbc_f, marker='o', markersize=8, color='purple', linestyle='-.', label='SBC,FACB')
plt.plot(ue_counts, y_tp_tr_f, marker='^', markersize=8, color='magenta', linestyle=':', label='TRA,FACB')
plt.title('UE Quantity vs Average Throughput')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Average Throughput (packets/slot)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# --- 右圖：Success Rate 比較 ---
plt.subplot(1, 2, 2)
plt.plot(ue_counts, y_sr_sbc_d, marker='s', markersize=8, color='green', label='SBC,DACB')
plt.plot(ue_counts, y_sr_tr_d, marker='d', markersize=8, color='orange', linestyle='--', label='TRA,DACB')
plt.plot(ue_counts, y_sr_sbc_f, marker='o', markersize=8, color='brown', linestyle='-.', label='SBC,FACB')
plt.plot(ue_counts, y_sr_tr_f, marker='^', markersize=8, color='cyan', linestyle=':', label='TRA,FACB')
plt.title('UE Quantity vs Success Rate')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Success Rate')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# 4. 優化佈局並顯示
plt.tight_layout()
plt.show()