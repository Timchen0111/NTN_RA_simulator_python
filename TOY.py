from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main


def format_group_label(group):
    return str(tuple(int(satellite_id) + 1 for satellite_id in group))


def summarize_group_tables(
    filename="group_ps_table.npz",
    preview_count=10,
    segment_size=100,
    segment_top_k=3,
    analysis_seconds=180,
):
    group_weight_table, group_ps_table, _ = main.load_ps_tables(filename)
    with np.load(filename, allow_pickle=True) as data:
        trao_ms = int(data["trao_ms"]) if "trao_ms" in data.files else 100

    # Use the same 3-minute window as the paper simulations.
    max_rao_count = analysis_seconds * 1000 // trao_ms
    used_rao_count = min(max_rao_count, len(group_weight_table))
    group_weight_table = group_weight_table[:used_rao_count]
    group_ps_table = group_ps_table[:used_rao_count]
    output_prefix = f"{Path(filename).stem}_first_{analysis_seconds}s"

    group_counts = np.array([len(groups) for groups in group_weight_table])
    histogram = Counter(group_counts.tolist())

    print("\n--- Group Count Summary ---")
    first_ps_table = next((table for table in group_ps_table if len(table) > 0), None)
    satellite_count = len(next(iter(first_ps_table.values()))) if first_ps_table is not None else 0
    print(f"Satellite count: {satellite_count}")
    print(f"Analysis window: first {analysis_seconds} seconds")
    print(f"RAO duration: {trao_ms} ms")
    print(f"RAO count: {len(group_counts)}")
    print(f"Average groups per RAO: {np.mean(group_counts):.4f}")
    print(f"Median groups per RAO: {np.median(group_counts):.4f}")
    print(f"Min groups per RAO: {np.min(group_counts)}")
    print(f"Max groups per RAO: {np.max(group_counts)}")

    print("\n--- Group Count Histogram ---")
    for count in sorted(histogram):
        print(f"{count} groups: {histogram[count]} RAOs")

    # Plot the number of preselection groups that exist in each RAO over time.
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(group_counts)), group_counts, linewidth=1.4)
    plt.title(f"Number of Preselection Groups per RAO")
    plt.xlabel("RAO Index")
    plt.ylabel("Group Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_group_count_per_rao.png", dpi=150)
    plt.show()

    print(f"\n--- First {preview_count} RAOs ---")
    for n in range(min(preview_count, len(group_weight_table))):
        print(f"RAO {n}: {len(group_weight_table[n])} groups")

    group_occurrences = Counter()
    for weights in group_weight_table:
        group_occurrences.update(weights.keys())

    most_common_groups = group_occurrences.most_common(preview_count)

    print(f"\n--- {preview_count} Most Common Groups ---")
    for group, count in most_common_groups:
        print(f"group {format_group_label(group)}: {count} RAOs")

    if most_common_groups:
        labels = [format_group_label(group) for group, _ in most_common_groups]
        counts = [count for _, count in most_common_groups]

        plt.figure(figsize=(12, 6))
        plt.bar(labels, counts)
        plt.title(f"Top {preview_count} Most Common Groups")
        plt.xlabel("Group")
        plt.ylabel("RAO Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_group_occurrence_histogram.png", dpi=150)
        plt.show()

        top_trend_groups = [group for group, _ in most_common_groups[:5]]

        plt.figure(figsize=(12, 5))
        for group in top_trend_groups:
            weight_trend = [
                group_weight_table[n].get(group, 0.0)
                for n in range(len(group_weight_table))
            ]
            plt.plot(range(len(weight_trend)), weight_trend, label=format_group_label(group))

        plt.title(f"Weight Trends for Top 5 Most Common Groups")
        plt.xlabel("RAO Index")
        plt.ylabel("w_g")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_top5_group_weight_trends.png", dpi=150)
        plt.show()

    integrated_group_weights = Counter()
    for weights in group_weight_table:
        integrated_group_weights.update(weights)

    average_group_weights = {
        group: total_weight / len(group_weight_table)
        for group, total_weight in integrated_group_weights.items()
    }
    top_integrated_groups = sorted(
        average_group_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:preview_count]

    print(f"\n--- Top {preview_count} Groups by Time-Averaged w_g ---")
    for group, avg_weight in top_integrated_groups:
        print(f"group {format_group_label(group)}: average w_g = {avg_weight:.6f}")

    if top_integrated_groups:
        top_group_set = {group for group, _ in top_integrated_groups}
        labels = [format_group_label(group) for group, _ in top_integrated_groups]
        values = [avg_weight for _, avg_weight in top_integrated_groups]
        other_weight = sum(
            avg_weight
            for group, avg_weight in average_group_weights.items()
            if group not in top_group_set
        )
        if other_weight > 0:
            labels.append("Others")
            values.append(other_weight)

        plt.figure(figsize=(12, 6))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Top {preview_count} Groups by Time-Averaged w_g")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_group_time_averaged_weight_ranking.png", dpi=150)
        plt.show()

    segment_records = []
    segment_top_groups = set()
    for start in range(0, len(group_weight_table), segment_size):
        end = min(start + segment_size, len(group_weight_table))
        segment_weights = Counter()
        for weights in group_weight_table[start:end]:
            segment_weights.update(weights)

        segment_length = end - start
        averaged_weights = {
            group: total_weight / segment_length
            for group, total_weight in segment_weights.items()
        }
        top_groups = sorted(
            averaged_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:segment_top_k]
        segment_records.append((start, end - 1, averaged_weights, top_groups))
        segment_top_groups.update(group for group, _ in top_groups)

    print(
        f"\n--- Segment Top {segment_top_k} Groups "
        f"(segment size = {segment_size} RAOs) ---"
    )
    for start, end, _, top_groups in segment_records:
        top_text = ", ".join(
            f"{format_group_label(group)}: {avg_weight:.4f}"
            for group, avg_weight in top_groups
        )
        print(f"RAO {start}-{end}: {top_text}")

    heatmap_groups = sorted(
        segment_top_groups,
        key=lambda group: average_group_weights.get(group, 0.0),
        reverse=True,
    )
    if heatmap_groups:
        heatmap = np.array([
            [
                averaged_weights.get(group, np.nan)
                if group in {top_group for top_group, _ in top_groups}
                else np.nan
                for _, _, averaged_weights, top_groups in segment_records
            ]
            for group in heatmap_groups
        ])
        x_labels = [f"{start}-{end}" for start, end, _, _ in segment_records]
        y_labels = [format_group_label(group) for group in heatmap_groups]

        fig_width = max(12, len(x_labels) * 0.7)
        fig_height = max(6, len(y_labels) * 0.35)
        plt.figure(figsize=(fig_width, fig_height))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        plt.imshow(np.ma.masked_invalid(heatmap), aspect="auto", cmap=cmap)
        plt.colorbar(label="Average group weight in each segment")
        plt.title("Weight distribution of the dominant groups")
        plt.xlabel("RAO Segment")
        plt.ylabel("Group")
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
        plt.yticks(range(len(y_labels)), y_labels)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_segment_top_group_heatmap.png", dpi=150)
        plt.show()

    return group_counts, group_weight_table, group_ps_table


if __name__ == "__main__":
    summarize_group_tables()
