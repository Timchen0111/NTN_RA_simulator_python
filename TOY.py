from collections import Counter

import numpy as np

import main


def summarize_group_tables(filename="group_ps_table.npz", preview_count=10):
    group_weight_table, group_ps_table = main.load_ps_tables(filename)

    group_counts = np.array([len(groups) for groups in group_weight_table])
    histogram = Counter(group_counts.tolist())

    print("\n--- Group Count Summary ---")
    print(f"RAO count: {len(group_counts)}")
    print(f"Average groups per RAO: {np.mean(group_counts):.4f}")
    print(f"Median groups per RAO: {np.median(group_counts):.4f}")
    print(f"Min groups per RAO: {np.min(group_counts)}")
    print(f"Max groups per RAO: {np.max(group_counts)}")

    print("\n--- Group Count Histogram ---")
    for count in sorted(histogram):
        print(f"{count} groups: {histogram[count]} RAOs")

    print(f"\n--- First {preview_count} RAOs ---")
    for n in range(min(preview_count, len(group_weight_table))):
        print(f"RAO {n}: {len(group_weight_table[n])} groups")

    print(f"\n--- {preview_count} RAOs With Fewest Groups ---")
    fewest_indices = np.argsort(group_counts)[:preview_count]
    for n in fewest_indices:
        print(f"RAO {int(n)}: {int(group_counts[n])} groups, weights={group_weight_table[n]}")

    return group_counts, group_weight_table, group_ps_table


if __name__ == "__main__":
    summarize_group_tables()
