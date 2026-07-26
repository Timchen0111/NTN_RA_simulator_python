import numpy as np

data = np.load("group_ps_table_top3.npz", allow_pickle=True)

weights = data["group_weight_table"]
ps_table = data["group_ps_table"]

print(weights[0])
print(ps_table[0])
print(data["group_size"])