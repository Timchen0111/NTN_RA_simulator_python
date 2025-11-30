import numpy as np

ue_pre = {'1': 6, '2': 9, '3': 5, '4': 9}
seen_values = set()
duplicates = set()
success_list = []
for value in ue_pre.values():
    if value in seen_values:
        duplicates.add(value)
    else:
        seen_values.add(value)
for ue in ue_pre.keys():
    if ue_pre[ue] not in duplicates:
        success_list.append(ue)
print(success_list)