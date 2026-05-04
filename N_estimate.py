import numpy as np

class RLSEstimator:
    def __init__(self, initial_N=0, P=10000, lam=0.995):
        """
        RLS 估測器初始化
        :param initial_N: N 的初始猜測值
        :param P: 初始協方差 (不確定性)，設定越大，初期更新越快
        :param lam: 遺忘因子 (Forgetting factor)，0.99~0.999 適合估計慢變量 N
        """
        self.N_tilde = initial_N
        self.P = P
        self.lam = lam
        self.epsilon = 1e-10  # 防止除以零

    def update(self, Lambda, denominator):
        """
        根據當前觀測到的 Lambda 與 數學推導的分母更新 N
        數學模型: Lambda = N * denominator
        """
        # phi 即為回歸量 (Regressor)
        phi = denominator
        
        # 1. 計算增益 K (Gain)
        # 這裡 phi 在分子，代表當 denominator 越小(資訊越差)，K 就會自動變小
        denom_part = self.lam + phi * self.P * phi
        K = (self.P * phi) / (denom_part + self.epsilon)
        
        # 2. 計算殘差 (Innovation / Error)
        # 比對「實際看到的流量」與「基於目前 N 預期應有的流量」
        error = Lambda - (phi * self.N_tilde)
        
        # 3. 更新參數 N_tilde
        self.N_tilde = self.N_tilde + K * error
        
        # 4. 更新協方差 P (Ricatti Equation)
        # P 會隨著時間收縮，代表系統越來越「有信心」，更新步長會自動變慢
        self.P = (1.0 / self.lam) * (self.P - K * phi * self.P)
        
        # 物理限制：總人數不應小於零
        self.N_tilde = max(0, self.N_tilde)
        
        return self.N_tilde

# --- 整合進你的模擬器 ---
'''
# 初始化 (在模擬開始前)
rls = RLSEstimator(initial_N=100, P=2000, lam=0.995)

# 在每個 Slot 更新 (取代原有的 N_estimation)
# Lambda: MoM 算出來的負載
# denominator: 你數學推導的 sum(pi_n * (1-pb))
current_N = rls.update(Lambda, denominator)
'''