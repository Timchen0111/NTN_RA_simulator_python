import main

group_weight_table, group_ps_table = main.load_ps_tables()
print("Group weight table sample:", group_weight_table[0])  # 打印第一個 RAO 的 group weight 表
print("Group PS table sample:", group_ps_table[1])  # 打印第一個