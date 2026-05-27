import main
import numpy as np
import matplotlib.pyplot as plt
import h5py


num_trials = 5  # 測試軌道面數量從 1 到 5
# 用於儲存每次實驗的結果指標
jaccard_record = np.zeros(num_trials)
cv_record = np.zeros(num_trials)
unique_sat_record = np.zeros(num_trials)

for i in range(num_trials):
    # 執行 main 函數並獲取評估結果
    # 假設 main 會回傳 evaluate_visibility_heterogeneity 的結果 dictionary
    result = main.main(0.01, i+1, 20, 10000, 0, 46, 1) #目前都是抓仰角10度以上的衛星，納入i個軌道面
    
    jaccard_record[i] = result['avg_jaccard']
    cv_record[i] = result['cv_visibility']
    unique_sat_record[i] = result['total_unique_sats']
    
    print(f"Orbit {i+2}: Jaccard={jaccard_record[i]:.4f}, CV={cv_record[i]:.4f}")

# --- 開始畫圖 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# 設定 X 軸 (軌道次數)
orbits = np.arange(2, num_trials + 2)

# 畫出 Jaccard Similarity (左側 Y 軸)
color = 'tab:blue'
ax1.set_xlabel('Orbit Number')
ax1.set_ylabel('Avg Jaccard Similarity', color=color)
ax1.plot(orbits, jaccard_record, color=color, marker='o', linestyle='-', label='Jaccard (Overlap)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# 建立右側 Y 軸用於畫 CV
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Visibility CV (Pressure)', color=color)
ax2.plot(orbits, cv_record, color=color, marker='s', linestyle='--', label='Visibility CV')
ax2.tick_params(axis='y', labelcolor=color)

# 加上標題
plt.title('Evaluation of Visibility Heterogeneity across Orbits\n(Spatial Asymmetry Assessment)')


fig.tight_layout()
plt.show()

# 額外儲存數據
print("\n--- Summary Statistics ---")
print(f"Mean Jaccard Similarity: {np.mean(jaccard_record):.4f} (Lower is more heterogeneous)")
print(f"Mean Visibility CV: {np.mean(cv_record):.4f} (Higher is more unbalanced)")
