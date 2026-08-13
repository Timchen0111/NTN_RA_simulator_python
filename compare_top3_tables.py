import argparse
from pathlib import Path

import numpy as np


DEFAULT_LEFT_TABLE = Path("group_ps_table_planes_2_top3.npz")
DEFAULT_RIGHT_TABLE = Path("group_ps_table_planes_4_top3.npz")


def scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if value.shape == () else value


def load_top3_table(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Top-3 table not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required_keys = (
            "group_weight_table",
            "group_ps_table",
            "sat_norad_ids",
            "rao_indices",
            "scenario_start_dt_iso",
            "tle_file_sha256",
        )
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise ValueError(f"{path} is missing required fields: {missing_keys}")

        table = {
            "path": path,
            "weights": data["group_weight_table"],
            "ps": data["group_ps_table"],
            "satellite_ids": np.asarray(data["sat_norad_ids"], dtype=int),
            "rao_indices": np.asarray(data["rao_indices"], dtype=int),
            "start": str(scalar(data, "scenario_start_dt_iso")),
            "tle_hash": str(scalar(data, "tle_file_sha256")),
            "plane_count": scalar(data, "orbit_plane_count", "not stored"),
            "group_size": scalar(data, "group_size", "not stored"),
            "seconds": scalar(data, "seconds", "not stored"),
            "trao_ms": scalar(data, "trao_ms", "not stored"),
            "num_points": scalar(data, "num_points", "not stored"),
            "rao_step": scalar(data, "rao_step", "not stored"),
            "radius_km": scalar(data, "radius_km", "not stored"),
            "random_seed": scalar(data, "random_seed", "not stored"),
        }

    if len(table["weights"]) != len(table["rao_indices"]):
        raise ValueError(f"{path} weight-table length does not match rao_indices.")
    if len(table["ps"]) != len(table["rao_indices"]):
        raise ValueError(f"{path} p_s-table length does not match rao_indices.")
    return table


def group_to_norad(group, satellite_ids):
    indices = np.asarray(tuple(group), dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(satellite_ids)):
        raise ValueError(f"Group contains an invalid satellite index: {tuple(group)}")
    return tuple(int(value) for value in satellite_ids[indices])


def weights_by_norad(weight_row, satellite_ids):
    return {
        group_to_norad(group, satellite_ids): float(weight)
        for group, weight in weight_row.items()
    }


def total_variation(left_weights, right_weights):
    all_groups = set(left_weights).union(right_weights)
    return 0.5 * sum(
        abs(left_weights.get(group, 0.0) - right_weights.get(group, 0.0))
        for group in all_groups
    )


def weighted_best_channel_probability(weight_row, ps_row):
    return float(sum(
        float(weight) * float(np.max(np.asarray(ps_row[group], dtype=float)))
        for group, weight in weight_row.items()
    ))


def marginal_channel_probability(weight_row, ps_row):
    first_group = next(iter(weight_row))
    satellite_count = len(np.asarray(ps_row[first_group], dtype=float))
    marginal = np.zeros(satellite_count, dtype=float)
    for group, weight in weight_row.items():
        marginal += float(weight) * np.asarray(ps_row[group], dtype=float)
    return marginal


def print_table_summary(table, label):
    group_counts = np.asarray([len(row) for row in table["weights"]], dtype=int)
    print(f"\n--- {label}: {table['path']} ---")
    print(f"Stored orbital planes: {table['plane_count']}")
    print(f"Satellites: {len(table['satellite_ids'])}")
    print(f"Sampled RAOs: {len(table['rao_indices'])}")
    print(f"RAO range: {table['rao_indices'][0]} to {table['rao_indices'][-1]}")
    print(f"Group size: {table['group_size']}")
    print(f"RAO step: {table['rao_step']}")
    print(f"Seconds / TRAO ms: {table['seconds']} / {table['trao_ms']}")
    print(f"UE samples / seed: {table['num_points']} / {table['random_seed']}")
    print(f"Radius: {table['radius_km']} km")
    print(f"Scenario start: {table['start']}")
    print(f"TLE hash: {table['tle_hash']}")
    print(
        "Distinct Top-3 groups per RAO: "
        f"mean={np.mean(group_counts):.3f}, "
        f"min={np.min(group_counts)}, max={np.max(group_counts)}"
    )


def compare_tables(left, right):
    print_table_summary(left, "Left")
    print_table_summary(right, "Right")

    left_ids = set(int(value) for value in left["satellite_ids"])
    right_ids = set(int(value) for value in right["satellite_ids"])
    common_ids = left_ids.intersection(right_ids)

    print("\n--- Satellite-pool comparison ---")
    print(f"Pools identical: {left_ids == right_ids}")
    print(f"Common satellites: {len(common_ids)}")
    print(f"Only in left: {len(left_ids - right_ids)} {sorted(left_ids - right_ids)}")
    print(f"Only in right: {len(right_ids - left_ids)} {sorted(right_ids - left_ids)}")

    print("\n--- Metadata comparison ---")
    print(f"Scenario start equal: {left['start'] == right['start']}")
    print(f"TLE hash equal: {left['tle_hash'] == right['tle_hash']}")
    print(
        "Sampled RAOs equal: "
        f"{np.array_equal(left['rao_indices'], right['rao_indices'])}"
    )
    for key in (
        "seconds",
        "trao_ms",
        "num_points",
        "rao_step",
        "radius_km",
        "random_seed",
    ):
        print(f"{key} equal: {left[key] == right[key]}")

    common_raos = np.intersect1d(left["rao_indices"], right["rao_indices"])
    left_row_by_rao = {
        int(rao): index for index, rao in enumerate(left["rao_indices"])
    }
    right_row_by_rao = {
        int(rao): index for index, rao in enumerate(right["rao_indices"])
    }

    count_equal = []
    group_sets_equal = []
    group_jaccard = []
    weight_tv = []
    best_channel_difference = []
    common_satellite_marginal_difference = []

    left_id_to_index = {
        int(norad_id): index
        for index, norad_id in enumerate(left["satellite_ids"])
    }
    right_id_to_index = {
        int(norad_id): index
        for index, norad_id in enumerate(right["satellite_ids"])
    }
    ordered_common_ids = sorted(common_ids)

    for rao in common_raos:
        left_index = left_row_by_rao[int(rao)]
        right_index = right_row_by_rao[int(rao)]
        left_weight_row = left["weights"][left_index]
        right_weight_row = right["weights"][right_index]
        left_ps_row = left["ps"][left_index]
        right_ps_row = right["ps"][right_index]

        left_weights = weights_by_norad(
            left_weight_row,
            left["satellite_ids"],
        )
        right_weights = weights_by_norad(
            right_weight_row,
            right["satellite_ids"],
        )
        left_groups = set(left_weights)
        right_groups = set(right_weights)
        union = left_groups.union(right_groups)

        count_equal.append(len(left_groups) == len(right_groups))
        group_sets_equal.append(left_groups == right_groups)
        group_jaccard.append(
            len(left_groups.intersection(right_groups)) / len(union)
            if union else 1.0
        )
        weight_tv.append(total_variation(left_weights, right_weights))
        best_channel_difference.append(
            weighted_best_channel_probability(right_weight_row, right_ps_row)
            - weighted_best_channel_probability(left_weight_row, left_ps_row)
        )

        left_marginal = marginal_channel_probability(left_weight_row, left_ps_row)
        right_marginal = marginal_channel_probability(
            right_weight_row,
            right_ps_row,
        )
        if ordered_common_ids:
            differences = [
                abs(
                    right_marginal[right_id_to_index[norad_id]]
                    - left_marginal[left_id_to_index[norad_id]]
                )
                for norad_id in ordered_common_ids
            ]
            common_satellite_marginal_difference.append(float(np.mean(differences)))

    count_equal = np.asarray(count_equal, dtype=bool)
    group_sets_equal = np.asarray(group_sets_equal, dtype=bool)
    group_jaccard = np.asarray(group_jaccard, dtype=float)
    weight_tv = np.asarray(weight_tv, dtype=float)
    best_channel_difference = np.asarray(best_channel_difference, dtype=float)

    print("\n--- Top-3 content comparison in NORAD-ID space ---")
    print(f"Common sampled RAOs: {len(common_raos)}")
    print(
        "RAOs with equal group counts: "
        f"{np.sum(count_equal)}/{len(count_equal)} "
        f"({np.mean(count_equal) * 100:.2f}%)"
    )
    print(
        "RAOs with identical group identities: "
        f"{np.sum(group_sets_equal)}/{len(group_sets_equal)} "
        f"({np.mean(group_sets_equal) * 100:.2f}%)"
    )
    print(
        "Group-set Jaccard similarity: "
        f"mean={np.mean(group_jaccard):.6f}, "
        f"min={np.min(group_jaccard):.6f}, "
        f"max={np.max(group_jaccard):.6f}"
    )
    print(
        "Group-weight total-variation distance: "
        f"mean={np.mean(weight_tv):.6f}, "
        f"min={np.min(weight_tv):.6f}, "
        f"max={np.max(weight_tv):.6f}"
    )
    print(
        "Right minus left weighted best-link p_s: "
        f"mean={np.mean(best_channel_difference):.6f}, "
        f"min={np.min(best_channel_difference):.6f}, "
        f"max={np.max(best_channel_difference):.6f}"
    )
    if common_satellite_marginal_difference:
        marginal_difference = np.asarray(
            common_satellite_marginal_difference,
            dtype=float,
        )
        print(
            "Mean absolute marginal-p_s difference over shared satellites: "
            f"mean={np.mean(marginal_difference):.6e}, "
            f"max={np.max(marginal_difference):.6e}"
        )

    contents_identical = (
        left_ids == right_ids
        and np.array_equal(left["rao_indices"], right["rao_indices"])
        and bool(np.all(group_sets_equal))
        and bool(np.allclose(weight_tv, 0.0, atol=0.0, rtol=0.0))
        and bool(np.allclose(
            best_channel_difference,
            0.0,
            atol=0.0,
            rtol=0.0,
        ))
    )
    print(f"\nTables identical by checked content: {contents_identical}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two ordered Top-3 preselection tables without modifying them."
        )
    )
    parser.add_argument(
        "left",
        nargs="?",
        type=Path,
        default=DEFAULT_LEFT_TABLE,
    )
    parser.add_argument(
        "right",
        nargs="?",
        type=Path,
        default=DEFAULT_RIGHT_TABLE,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    compare_tables(
        load_top3_table(arguments.left),
        load_top3_table(arguments.right),
    )
