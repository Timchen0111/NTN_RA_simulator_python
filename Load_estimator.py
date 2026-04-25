import numpy as np

def precompute_expected_tables(Z, Nmax=1000):
    """
    在模擬開始前預先計算期望值表
    Z: 每顆衛星的前導碼總數
    Nmax: 負載搜尋上限
    """
    E_i_table = np.zeros(Nmax)
    E_s_table = np.zeros(Nmax)
    E_c_table = np.zeros(Nmax)
    
    for i in range(Nmax):
        # 1. 閒置期望值 E_i
        E_i_table[i] = Z * ((1 - 1/Z)**i)
        
        # 2. 成功期望值 E_s
        if i > 0:
            E_s_table[i] = i * ((1 - 1/Z)**(i-1))
        else:
            E_s_table[i] = 0
            
        # 3. 碰撞期望值 E_c
        E_c_table[i] = Z - E_i_table[i] - E_s_table[i]
        
    return E_i_table, E_s_table, E_c_table

def load_estimator(N_i, N_s, N_c, tables):
    """
    實作高度優化的 MoM 負載估計器
    tables: 傳入預計算好的 (E_i_table, E_s_table, E_c_table)
    """
    K = len(N_i)
    Lambda = np.zeros(K)
    E_i_table, E_s_table, E_c_table = tables
    Nmax = len(E_i_table)
    
    for k in range(K):
        min_error = 1e9
        best_i = 0
        
        # 純查表比對，不涉及任何指數或乘除運算
        for i in range(Nmax):
            error = abs(N_i[k] - E_i_table[i]) + \
                    abs(N_s[k] - E_s_table[i]) + \
                    abs(N_c[k] - E_c_table[i])
            
            if error < min_error:
                min_error = error
                best_i = i
        
        Lambda[k] = best_i
        
    return Lambda