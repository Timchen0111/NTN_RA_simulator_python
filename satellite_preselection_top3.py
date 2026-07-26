from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs

from main import estimate_channel_success_probability, load_fixed_satellites
from satellite_preselection import generate_uniform_locations
from scenario_time import get_tle_scenario_metadata


REFERENCE_TABLE = Path("group_ps_table.npz")
OUTPUT_TABLE = Path("group_ps_table_top3.npz")
FIXED_SATELLITE_POOL = "fixed_satellite_pool.json"

RANDOM_SEED = 42
CENTER = (25.03, 121.56)
RADIUS_KM = 200.0
GROUP_SIZE = 3

# False: only calculate RAO 0, 10, 20, ... for snapshot analysis.
# True: calculate every RAO so the table can be indexed like the original table.
GENERATE_FULL_TABLE = False
SAMPLED_RAO_STEP = 10


def prepare_ue_geometry(sample_locations):
    """Convert fixed UE locations to the ECEF/ENU arrays used by batch geometry."""
    ue_ecef_km = np.vstack([location.itrs_xyz.km for location in sample_locations])
    lat_rad = np.deg2rad([location.latitude.degrees for location in sample_locations])
    lon_rad = np.deg2rad([location.longitude.degrees for location in sample_locations])

    east = np.column_stack((
        -np.sin(lon_rad),
        np.cos(lon_rad),
        np.zeros(len(sample_locations)),
    ))
    north = np.column_stack((
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad),
    ))
    up = np.column_stack((
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ))
    return ue_ecef_km, east, north, up


def compute_top3_group_ps_table(
    real_sats,
    start_dt,
    seconds,
    trao_ms,
    sample_locations,
    filename=OUTPUT_TABLE,
    scenario_metadata=None,
    generate_full_table=GENERATE_FULL_TABLE,
    sampled_rao_step=SAMPLED_RAO_STEP,
):
    """Generate ordered Top-3 group weights and per-satellite channel success rates."""
    output_path = Path(filename)
    if output_path.resolve() == REFERENCE_TABLE.resolve():
        raise ValueError(
            "Refusing to overwrite group_ps_table.npz; use a separate Top-3 filename."
        )
    if len(real_sats) < GROUP_SIZE:
        raise ValueError(f"Top-{GROUP_SIZE} grouping requires at least {GROUP_SIZE} satellites.")

    if scenario_metadata is None:
        scenario_metadata = get_tle_scenario_metadata()

    ts = load.timescale()
    full_rao_count = seconds * 1000 // trao_ms
    if sampled_rao_step <= 0:
        raise ValueError("sampled_rao_step must be positive.")
    rao_step = 1 if generate_full_table else sampled_rao_step
    rao_indices = np.arange(0, full_rao_count, rao_step, dtype=int)
    num_sat = len(real_sats)
    num_points = len(sample_locations)
    ue_ecef_km, east, north, up = prepare_ue_geometry(sample_locations)

    group_weight_table = []
    group_ps_table = []

    for table_index, n in enumerate(rao_indices):
        n = int(n)
        current_dt = start_dt + timedelta(milliseconds=n * trao_ms)
        current_t = ts.from_datetime(current_dt)

        sat_ecef_km = np.stack(
            [sat.at(current_t).frame_xyz(itrs).km for sat in real_sats],
            axis=0,
        )
        delta = sat_ecef_km[None, :, :] - ue_ecef_km[:, None, :]
        up_component = np.einsum("nkd,nd->nk", delta, up)
        east_component = np.einsum("nkd,nd->nk", delta, east)
        north_component = np.einsum("nkd,nd->nk", delta, north)
        horizontal_distance = np.hypot(east_component, north_component)

        angles = np.degrees(np.arctan2(up_component, horizontal_distance))
        distances = np.linalg.norm(delta, axis=2)
        ps_matrix = estimate_channel_success_probability(angles, distances)
        top3_indices = np.argsort(angles, axis=1)[:, ::-1][:, :GROUP_SIZE]

        group_count = defaultdict(int)
        group_ps_sum = {}
        for point_index, top3 in enumerate(top3_indices):
            group = tuple(int(satellite_id) for satellite_id in top3)
            group_count[group] += 1
            if group not in group_ps_sum:
                group_ps_sum[group] = np.zeros(num_sat)
            group_ps_sum[group] += ps_matrix[point_index]

        weights = {
            group: count / num_points
            for group, count in group_count.items()
        }
        ps_by_group = {
            group: group_ps_sum[group] / group_count[group]
            for group in group_count
        }
        group_weight_table.append(weights)
        group_ps_table.append(ps_by_group)

        if table_index % 10 == 0:
            print(
                f"RAO {n}/{full_rao_count}: "
                f"Top-3 groups = {len(weights)}"
            )

    np.savez_compressed(
        output_path,
        group_weight_table=np.array(group_weight_table, dtype=object),
        group_ps_table=np.array(group_ps_table, dtype=object),
        sat_norad_ids=np.array([int(sat.model.satnum) for sat in real_sats]),
        scenario_start_dt_iso=scenario_metadata["start_dt_iso"],
        tle_epoch_min_iso=scenario_metadata["tle_epoch_min_iso"],
        tle_epoch_max_iso=scenario_metadata["tle_epoch_max_iso"],
        tle_epoch_median_iso=scenario_metadata["tle_epoch_median_iso"],
        tle_file_sha256=scenario_metadata["tle_file_sha256"],
        seconds=seconds,
        trao_ms=trao_ms,
        num_points=num_points,
        rao_indices=rao_indices,
        full_rao_count=full_rao_count,
        rao_step=rao_step,
        is_full_table=generate_full_table,
        group_size=GROUP_SIZE,
        random_seed=RANDOM_SEED,
        center_lat=CENTER[0],
        center_lon=CENTER[1],
        radius_km=RADIUS_KM,
    )
    print(f"Saved Top-3 group p_s table to {output_path}")


def main():
    if not REFERENCE_TABLE.exists():
        raise FileNotFoundError(
            "group_ps_table.npz is required as the reference configuration."
        )

    reference = np.load(REFERENCE_TABLE, allow_pickle=True)
    seconds = int(reference["seconds"])
    trao_ms = int(reference["trao_ms"])
    num_points = int(reference["num_points"])
    expected_sat_ids = np.asarray(reference["sat_norad_ids"], dtype=int)

    scenario_metadata = get_tle_scenario_metadata()
    if str(reference["scenario_start_dt_iso"]) != scenario_metadata["start_dt_iso"]:
        raise ValueError("Reference table and current scenario start time do not match.")
    if str(reference["tle_file_sha256"]) != scenario_metadata["tle_file_sha256"]:
        raise ValueError("Reference table and current TLE file do not match.")

    real_sats = load_fixed_satellites(FIXED_SATELLITE_POOL)
    actual_sat_ids = np.array([int(sat.model.satnum) for sat in real_sats])
    if not np.array_equal(actual_sat_ids, expected_sat_ids):
        raise ValueError(
            "fixed_satellite_pool.json does not match the reference group_ps_table.npz."
        )

    np.random.seed(RANDOM_SEED)
    sample_locations = generate_uniform_locations(
        num_points=num_points,
        center=CENTER,
        R_km=RADIUS_KM,
    )

    compute_top3_group_ps_table(
        real_sats=real_sats,
        start_dt=scenario_metadata["start_dt"],
        seconds=seconds,
        trao_ms=trao_ms,
        sample_locations=sample_locations,
        filename=OUTPUT_TABLE,
        scenario_metadata=scenario_metadata,
        generate_full_table=GENERATE_FULL_TABLE,
        sampled_rao_step=SAMPLED_RAO_STEP,
    )


if __name__ == "__main__":
    main()
