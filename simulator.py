import main
import matplotlib.pyplot as plt
results = []
plot_num = 5
for i in range(plot_num):
    r = main.main(100*(i+1),2,100)
    results.append(r)
print(results)

# 1. 準備 X 軸數據 (UE 數量)
x_ue = [100 * (i + 1) for i in range(plot_num)]

# 2. 準備 Y 軸數據 (從 results 裡面拆解出來)
# results 裡的格式是 [(throughput, rate), (throughput, rate), ...]
y_throughput = [r[0] for r in results]
y_success_rate = [r[1] for r in results]

# 3. 設定畫布大小
plt.figure(figsize=(12, 5))

# --- 第一張圖：Average Throughput ---
plt.subplot(1, 2, 1)  # 1列2行，這是第1張
plt.plot(x_ue, y_throughput, marker='o', color='blue', label='Avg Throughput')
plt.title('UE Quantity vs Average Throughput')
plt.xlabel('UE Quantity')
plt.ylabel('Average Throughput')
plt.grid(True)
plt.legend()

# --- 第二張圖：Successful Rate ---
plt.subplot(1, 2, 2)  # 1列2行，這是第2張
plt.plot(x_ue, y_success_rate, marker='s', color='green', label='Success Rate')
plt.title('UE Quantity vs Successful Rate')
plt.xlabel('UE Quantity')
plt.ylabel('Successful Rate')
plt.grid(True)
plt.legend()

# 4. 顯示圖表
plt.tight_layout() # 避免標籤重疊
plt.show()

    