from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs

from main import estimate_channel_success_probability, load_fixed_satellites
from satellite_preselection import generate_uniform_locations
from scenario_time import get_tle_scenario_metadata


TOP3_SCENARIOS = (
    (
        1,
        Path("group_ps_table_planes_1.npz"),
        "fixed_satellite_pool_planes_1.json",
        Path("group_ps_table_planes_1_top3.npz"),
    ),
    (
        2,
        Path("group_ps_table_planes_2.npz"),
        "fixed_satellite_pool_planes_2.json",
        Path("group_ps_table_planes_2_top3.npz"),
    ),
    (
        4,
        Path("group_ps_table_planes_4.npz"),
        "fixed_satellite_pool_planes_4.json",
        Path("group_ps_table_planes_4_top3.npz"),
    ),
)

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
    filename,
    reference_filename,
    orbit_plane_count,
    scenario_metadata=None,
    generate_full_table=GENERATE_FULL_TABLE,
    sampled_rao_step=SAMPLED_RAO_STEP,
):
    """Generate ordered Top-3 group weights and per-satellite channel success rates."""
    output_path = Path(filename)
    reference_path = Path(reference_filename)
    if output_path.resolve() == reference_path.resolve():
        raise ValueError(
            f"Refusing to overwrite reference table {reference_path}."
        )
    if output_path.exists():
        raise FileExistsError(
            f"Top-3 output already exists; refusing to overwrite: {output_path}"
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
        orbit_plane_count=orbit_plane_count,
        source_reference_table=reference_path.name,
    )
    print(f"Saved Top-3 group p_s table to {output_path}")


def generate_top3_scenario(
    orbit_plane_count,
    reference_table,
    fixed_satellite_pool,
    output_table,
):
    reference_table = Path(reference_table)
    output_table = Path(output_table)
    fixed_satellite_pool = Path(fixed_satellite_pool)

    if not reference_table.exists():
        raise FileNotFoundError(
            f"Reference table not found: {reference_table}"
        )
    if not fixed_satellite_pool.exists():
        raise FileNotFoundError(
            f"Fixed satellite pool not found: {fixed_satellite_pool}"
        )
    if output_table.exists():
        raise FileExistsError(
            f"Top-3 output already exists; refusing to overwrite: {output_table}"
        )

    with np.load(reference_table, allow_pickle=True) as reference:
        required_keys = (
            "seconds",
            "trao_ms",
            "num_points",
            "sat_norad_ids",
            "scenario_start_dt_iso",
            "tle_file_sha256",
        )
        missing_keys = [
            key for key in required_keys if key not in reference.files
        ]
        if missing_keys:
            raise ValueError(
                f"{reference_table} is missing required metadata: "
                f"{missing_keys}"
            )
        seconds = int(reference["seconds"])
        trao_ms = int(reference["trao_ms"])
        num_points = int(reference["num_points"])
        expected_sat_ids = np.asarray(
            reference["sat_norad_ids"],
            dtype=int,
        )
        reference_start_dt_iso = str(reference["scenario_start_dt_iso"])
        reference_tle_hash = str(reference["tle_file_sha256"])
        if "radius_km" in reference.files:
            reference_radius_km = float(reference["radius_km"])
            if not np.isclose(reference_radius_km, RADIUS_KM):
                raise ValueError(
                    f"{reference_table} radius is {reference_radius_km:g} km; "
                    f"expected {RADIUS_KM:g} km."
                )

    scenario_metadata = get_tle_scenario_metadata()
    if reference_start_dt_iso != scenario_metadata["start_dt_iso"]:
        raise ValueError(
            f"{reference_table} and current scenario start time do not match."
        )
    if reference_tle_hash != scenario_metadata["tle_file_sha256"]:
        raise ValueError(
            f"{reference_table} and current TLE file do not match."
        )

    real_sats = load_fixed_satellites(str(fixed_satellite_pool))
    actual_sat_ids = np.array([int(sat.model.satnum) for sat in real_sats])
    if not np.array_equal(actual_sat_ids, expected_sat_ids):
        raise ValueError(
            f"{fixed_satellite_pool} does not match {reference_table}."
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
        filename=output_table,
        reference_filename=reference_table,
        orbit_plane_count=orbit_plane_count,
        scenario_metadata=scenario_metadata,
        generate_full_table=GENERATE_FULL_TABLE,
        sampled_rao_step=SAMPLED_RAO_STEP,
    )


if __name__ == "__main__":
    missing_inputs = [
        str(path)
        for _, reference_table, fixed_satellite_pool, _ in TOP3_SCENARIOS
        for path in (reference_table, Path(fixed_satellite_pool))
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot generate the Top-3 tables; missing Mode 17 inputs: "
            + ", ".join(missing_inputs)
        )

    existing_outputs = [
        str(output_table)
        for _, _, _, output_table in TOP3_SCENARIOS
        if output_table.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing Top-3 outputs: "
            + ", ".join(existing_outputs)
        )

    for scenario in TOP3_SCENARIOS:
        orbit_plane_count = scenario[0]
        print(
            f"\n=== Generating Top-3 preselection data for "
            f"{orbit_plane_count} orbital plane(s) ==="
        )
        generate_top3_scenario(*scenario)
