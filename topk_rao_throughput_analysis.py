import csv
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs

from main import estimate_channel_success_probability, load_fixed_satellites
from satellite_preselection import generate_uniform_locations
from satellite_preselection_top3 import prepare_ue_geometry
from scenario_time import get_tle_scenario_metadata
from selection import solve_group_selection_policy


TABLE_FILE = "group_ps_table_top3.npz"
OUTPUT_CSV = "topk_ss_channel_success_all_rao.csv"
OUTPUT_FIGURE = "topk_ss_channel_success_all_rao.png"
OUTPUT_PDF = Path("output/pdf/topk_ss_channel_success_all_rao.pdf")
SEED = 42
EPSILON = 0.001
POLICIES = (("Global", 0), ("Top-1", 1), ("Top-2", 2), ("Top-3", 3))
warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)


def get_ue_channel_data(real_sats, current_time, ue_geometry):
    ecef, east, north, up = ue_geometry
    sat_ecef = np.stack([
        sat.at(current_time).frame_xyz(itrs).km for sat in real_sats
    ])
    delta = sat_ecef[None, :, :] - ecef[:, None, :]
    up_component = np.einsum("nkd,nd->nk", delta, up)
    east_component = np.einsum("nkd,nd->nk", delta, east)
    north_component = np.einsum("nkd,nd->nk", delta, north)
    angles = np.degrees(np.arctan2(
        up_component, np.hypot(east_component, north_component)
    ))
    distances = np.linalg.norm(delta, axis=2)
    channel_ps = estimate_channel_success_probability(angles, distances)
    ranking = np.argsort(angles, axis=1)[:, ::-1]
    return channel_ps, ranking


def merge_groups(weights3, ps3, prefix_length):
    weights, weighted_ps = {}, {}
    for group3, weight in weights3.items():
        group = tuple(group3[:prefix_length]) if prefix_length else ()
        weights[group] = weights.get(group, 0.0) + weight
        weighted_ps.setdefault(group, np.zeros_like(ps3[group3]))
        weighted_ps[group] += weight * ps3[group3]
    ps = {group: weighted_ps[group] / weights[group] for group in weights}
    return weights, ps


def evaluate_policy(
    policy, prefix_length, weights, group_ps, channel_ps, ranking,
    selection_uniforms, channel_uniforms,
):
    num_ues, num_sats = channel_ps.shape
    selection_ps = np.zeros((num_ues, num_sats))
    fallback_count = 0

    for ue_index in range(num_ues):
        group = tuple(ranking[ue_index, :prefix_length]) if prefix_length else ()
        if group in policy:
            selection_ps[ue_index] = policy[group]
        else:
            selection_ps[ue_index, ranking[ue_index, 0]] = 1.0
            fallback_count += 1

    # Exact objective value calculated from the group-level optimization inputs.
    effective_load = sum(
        weights[group] * policy[group] * group_ps[group] for group in weights
    )
    predicted_ps = float(np.sum(effective_load))
    imbalance = float(np.sum((effective_load - predicted_ps / num_sats) ** 2))

    # Exact expectation using every UE's own location and channel probabilities.
    per_ue_expected_ps = float(np.mean(np.sum(selection_ps * channel_ps, axis=1)))

    # Per-UE satellite selection followed by one channel-model realization.
    selected_sats = np.sum(
        selection_uniforms[:, None] > np.cumsum(selection_ps, axis=1), axis=1
    )
    selected_sats = np.minimum(selected_sats, num_sats - 1)
    ue_indices = np.arange(num_ues)
    channel_passed = (
        channel_uniforms[ue_indices, selected_sats]
        < channel_ps[ue_indices, selected_sats]
    )
    return {
        "expected_ps": predicted_ps,
        "per_ue_expected_ps": per_ue_expected_ps,
        "simulated_channel_success_rate": float(np.mean(channel_passed)),
        "imbalance": imbalance,
        "fallbacks": fallback_count,
    }


def save_tradeoff_figure(results):
    names = [name for name, _ in POLICIES]
    mean_ps = np.array([
        np.mean([
            float(row["expected_ps"]) for row in results if row["policy"] == name
        ])
        for name in names
    ])
    mean_groups = np.array([
        np.mean([
            float(row["groups"]) for row in results if row["policy"] == name
        ])
        for name in names
    ])

    x = np.arange(len(names))
    width = 0.36
    figure, success_axis = plt.subplots(figsize=(9.5, 5.5), dpi=140)
    group_axis = success_axis.twinx()

    success_bars = success_axis.bar(
        x - width / 2, mean_ps, width,
        color="#4C78A8", alpha=0.9,
        label="Preamble transmission success probability",
    )
    success_axis.bar_label(success_bars, fmt="%.4f", padding=3)
    success_axis.set_ylabel("Preamble transmission success probability")
    success_axis.set_ylim(0, 0.75)

    group_bars = group_axis.bar(
        x + width / 2, mean_groups, width,
        color="#6B7280", alpha=0.9,
        label="Average number of groups",
    )
    group_axis.bar_label(group_bars, fmt="%.1f", padding=3)
    group_axis.set_ylabel("Average number of groups")
    group_axis.set_ylim(0, 40)

    success_axis.set(
        title="Preamble Transmission Success Probability under Different Grouping Policies",
        xlabel="Grouping policy",
        xticks=x,
        xticklabels=names,
    )
    success_axis.grid(axis="y", alpha=0.25)
    success_axis.set_axisbelow(True)
    success_axis.legend(
        [success_bars, group_bars],
        ["Preamble transmission success probability", "Average number of groups"],
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)


def main():
    data = np.load(TABLE_FILE, allow_pickle=True)
    weight_table, ps_table = data["group_weight_table"], data["group_ps_table"]
    rao_indices, trao_ms = data["rao_indices"], int(data["trao_ms"])
    num_ues = int(data["num_points"])
    table_seed = int(data["random_seed"])
    center = (float(data["center_lat"]), float(data["center_lon"]))
    radius_km = float(data["radius_km"])
    real_sats = load_fixed_satellites()
    sat_ids = np.array([int(sat.model.satnum) for sat in real_sats])
    if not np.array_equal(sat_ids, data["sat_norad_ids"]):
        raise ValueError("Satellite pool does not match the Top-3 table.")

    np.random.seed(table_seed)
    locations = generate_uniform_locations(num_ues, center, radius_km)
    ue_geometry = prepare_ue_geometry(locations)
    scenario = get_tle_scenario_metadata()
    timescale = load.timescale()
    rng = np.random.default_rng(SEED)
    rows = np.arange(len(rao_indices))
    results = []

    for row in rows:
        actual_rao = int(rao_indices[row])
        current_dt = scenario["start_dt"] + timedelta(
            milliseconds=actual_rao * trao_ms
        )
        current_time = timescale.from_datetime(current_dt)
        channel_ps, ranking = get_ue_channel_data(
            real_sats, current_time, ue_geometry
        )
        selection_uniforms = rng.random(num_ues)
        channel_uniforms = rng.random(channel_ps.shape)

        for name, prefix_length in POLICIES:
            weights, group_ps = merge_groups(
                weight_table[row], ps_table[row], prefix_length
            )
            policy = solve_group_selection_policy(
                weights, group_ps, sat_num=len(real_sats),
                imbalance_epsilon=EPSILON, initial_policy=None,
            )
            metrics = evaluate_policy(
                policy, prefix_length, weights, group_ps, channel_ps, ranking,
                selection_uniforms, channel_uniforms,
            )
            results.append({
                "rao": actual_rao,
                "policy": name,
                "groups": len(weights),
                **metrics,
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)

    names = [name for name, _ in POLICIES]
    predicted = [
        np.array([r["expected_ps"] for r in results if r["policy"] == name])
        for name in names
    ]
    simulated = [
        np.array([
            r["simulated_channel_success_rate"]
            for r in results if r["policy"] == name
        ])
        for name in names
    ]
    save_tradeoff_figure(results)

    print(
        f"Analyzed all {len(rows)} table RAOs: "
        f"{int(rao_indices[rows[0]])} to {int(rao_indices[rows[-1]])}"
    )
    for name, predicted_values, simulated_values in zip(
        names, predicted, simulated
    ):
        policy_rows = [r for r in results if r["policy"] == name]
        print(
            f"{name}: expected p_s={predicted_values.mean():.6f}, "
            f"per-UE simulation={simulated_values.mean():.6f}, "
            f"mean groups={np.mean([r['groups'] for r in policy_rows]):.1f}, "
            f"max imbalance={max(r['imbalance'] for r in policy_rows):.6g}, "
            f"fallbacks={sum(r['fallbacks'] for r in policy_rows)}"
        )


if __name__ == "__main__":
    main()
