import numpy as np
import Load_estimator as l
import old as main
import matplotlib.pyplot as plt

def test_mom_performance():
    # --- 1. 設定測試參數 ---
    Z = 54           # 前導碼總數
    Nmax = 1000       # 估計搜尋上限
    test_lambdas = np.arange(1, 302, 10)  # 修改點：測試負載從 1 到 301，每隔 10 點取樣
    num_trials = 50   # 每個負載點重複實驗次數
    
    # 預計算理論表
    tables = l.precompute_expected_tables(Z, Nmax)
    
    ground_truth = []
    estimations = []

    print(f"Start to test MoM Estimator (Z={Z}, Nmax={Nmax})...")

    # --- 2. 執行模擬測試 ---
    for true_lambda in test_lambdas:
        trial_estimates = []
        for _ in range(num_trials):
            selections = np.random.randint(0, Z, size=true_lambda)
            
            counts = np.bincount(selections, minlength=Z)
            n_i = np.sum(counts == 0)
            n_s = np.sum(counts == 1)
            n_c = np.sum(counts > 1)
            
            hat_lambda = l.load_estimator(np.array([n_i]), np.array([n_s]), np.array([n_c]), tables)
            trial_estimates.append(hat_lambda[0])
            
        ground_truth.append(true_lambda)
        estimations.append(np.mean(trial_estimates))

    # --- 3. 計算誤差指標 ---
    ground_truth = np.array(ground_truth)
    estimations = np.array(estimations)
    mae = np.mean(np.abs(ground_truth - estimations))
    print(f"Complete! MAE: {mae:.2f}")

    # 計算每個測試點的絕對誤差 (Absolute Error)
    abs_errors = np.abs(estimations - ground_truth)

    # --- 設定雙 X 軸需要的整數比例與對應位置 ---
    # 修改點：配合 301 上限，整數比例設定到 5.0 即可 (5 * 54 = 270)
    desired_ratios = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    corresponding_lambdas = desired_ratios * Z

    # --- 4. 繪製合併圖表 (2 行 1 列) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

    # ------------------ 上圖：估計值與真實值對照 ------------------
    ax1.plot(ground_truth, ground_truth, 'r--', label='Ideal (Ground Truth)')
    ax1.scatter(ground_truth, estimations, color='blue', label='MoM Estimation')
    ax1.set_ylabel('Estimated Load (Lambda Hat)')
    ax1.set_title('MoM Estimator Performance Analysis', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=-10, right=315)  # 修改點：調整 X 軸邊界以符合 301

    # 上圖的上方整數比例軸
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(corresponding_lambdas)
    ax1_top.set_xticklabels([f'{r:.1f}' for r in desired_ratios])
    ax1_top.set_xlabel('Load Ratio (True Lambda / Z)')
    
    # 加上垂直輔助線
    for pos in corresponding_lambdas:
        ax1.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)


    # ------------------ 下圖：估計絕對誤差 ------------------
    ax2.plot(ground_truth, abs_errors, color='purple', marker='o', linewidth=1.5, label='Absolute Error')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.6, label='Zero Error')
    ax2.set_xlabel('True Load (Number of UEs)')
    ax2.set_ylabel('Absolute Error |Estimated - True|')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(ax1.get_xlim())  # 確保上下圖的 X 軸範圍完全一致

    # 下圖的上方整數比例軸
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(corresponding_lambdas)
    ax2_top.set_xticklabels([f'{r:.1f}' for r in desired_ratios])
    
    # 加上垂直輔助線
    for pos in corresponding_lambdas:
        ax2.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

test_mom_performance()