import main
import matplotlib.pyplot as plt

# 1. 執行模擬並收集數據
#ue_counts = [50,100,150]
ue_counts = [50] # 測試用，正式跑的時候改回上面那行
results_proposed = []
results_traditional = []
results_fixed_acb = []

for count in ue_counts:
    r_prop = main.main(count, 1, 100, 2)
    r_fixed_acb = main.main(count, 1, 100, 1)
    r_trad = main.main(count, 1, 100, 0)
        
    results_proposed.append(r_prop)
    results_traditional.append(r_trad)
    results_fixed_acb.append(r_fixed_acb)


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
plt.plot(ue_counts, y_tp_prop, marker='s', markersize=8, color='blue', label='Proposed (SBC)')
plt.plot(ue_counts, y_tp_trad, marker='d', markersize=8, color='red', linestyle='--', label='Traditional (Single ACB test)')
plt.plot(ue_counts, y_tp_fixed_acb, marker='o', markersize=8, color='purple', linestyle='-.', label='Fixed ACB (0.5)')
plt.title('UE Quantity vs Average Throughput')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Average Throughput (packets/slot)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# --- 右圖：Success Rate 比較 ---
plt.subplot(1, 2, 2)
plt.plot(ue_counts, y_sr_prop, marker='s', markersize=8, color='green', label='Proposed (SBC)')
plt.plot(ue_counts, y_sr_trad, marker='d', markersize=8, color='orange', linestyle='--', label='Traditional (Single ACB test)')
plt.plot(ue_counts, y_sr_fixed_acb, marker='o', markersize=8, color='brown', linestyle='-.', label='Fixed ACB (0.5)')
plt.title('UE Quantity vs Success Rate')
plt.xlabel('UE Quantity (Massive IoT Devices)')
plt.ylabel('Success Rate')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# 4. 優化佈局並顯示
plt.tight_layout()
plt.show()

    