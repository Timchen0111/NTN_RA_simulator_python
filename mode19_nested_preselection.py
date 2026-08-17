import argparse
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
from skyfield.api import load, wgs84

from main import channel_visibility
from satellite_preselection import generate_uniform_locations
from satellite_preselection_top3 import (
    CENTER,
    GROUP_SIZE,
    RANDOM_SEED,
    RADIUS_KM,
    compute_top3_group_ps_table,
)
from scenario_time import get_tle_scenario_metadata, load_starlink_tles


BASE_POOL = Path("fixed_satellite_pool_planes_2.json")
TEMPLATE_TABLE = Path("group_ps_table_planes_2_top3.npz")

NESTED_3_POOL = Path("fixed_satellite_pool_planes_3_nested.json")
NESTED_4_POOL = Path("fixed_satellite_pool_planes_4_nested.json")
NESTED_3_TABLE = Path("group_ps_table_planes_3_nested_top3.npz")
NESTED_4_TABLE = Path("group_ps_table_planes_4_nested_top3.npz")

# Representatives recovered from the tested orbit.py ranking.
# Rank 3: 10 visible seed satellites.
# Rank 8: 8 visible seed satellites and the best useful fourth-plane candidate.
PLANE_3_REPRESENTATIVE_NORAD = 66386
PLANE_4_REPRESENTATIVE_NORAD = 57634

INCLINATION_TOLERANCE_DEG = 1.0
RAAN_TOLERANCE_DEG = 5.0
MIN_ELEVATION_DEG = 10.0


def scalar(data, key):
    value = data[key]
    return value.item() if value.shape == () else value


def circular_angle_difference_deg(left, right):
    return abs((left - right + 180.0) % 360.0 - 180.0)


def satellite_orbit_values(satellite):
    inclination_deg = float(np.degrees(satellite.model.inclo))
    raan_deg = float(np.degrees(satellite.model.nodeo) % 360.0)
    return inclination_deg, raan_deg


def load_template_metadata(filename):
    required_keys = (
        "sat_norad_ids",
        "scenario_start_dt_iso",
        "tle_file_sha256",
        "seconds",
        "trao_ms",
        "num_points",
        "rao_indices",
        "rao_step",
        "group_size",
        "random_seed",
        "center_lat",
        "center_lon",
        "radius_km",
    )
    with np.load(filename, allow_pickle=True) as data:
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise ValueError(
                f"{filename} is missing required fields: {missing_keys}"
            )
        metadata = {
            "satellite_ids": np.asarray(data["sat_norad_ids"], dtype=int),
            "scenario_start_dt_iso": str(
                scalar(data, "scenario_start_dt_iso")
            ),
            "tle_file_sha256": str(scalar(data, "tle_file_sha256")),
            "seconds": int(scalar(data, "seconds")),
            "trao_ms": int(scalar(data, "trao_ms")),
            "num_points": int(scalar(data, "num_points")),
            "rao_indices": np.asarray(data["rao_indices"], dtype=int),
            "rao_step": int(scalar(data, "rao_step")),
            "group_size": int(scalar(data, "group_size")),
            "random_seed": int(scalar(data, "random_seed")),
            "center_lat": float(scalar(data, "center_lat")),
            "center_lon": float(scalar(data, "center_lon")),
            "radius_km": float(scalar(data, "radius_km")),
        }

    if metadata["group_size"] != GROUP_SIZE:
        raise ValueError(
            f"{filename} uses Top-{metadata['group_size']}; "
            f"expected Top-{GROUP_SIZE}."
        )
    if metadata["random_seed"] != RANDOM_SEED:
        raise ValueError(
            f"{filename} random seed is {metadata['random_seed']}; "
            f"expected {RANDOM_SEED}."
        )
    if not np.isclose(metadata["center_lat"], CENTER[0]) or not np.isclose(
        metadata["center_lon"], CENTER[1]
    ):
        raise ValueError(f"{filename} uses a different service-area center.")
    if not np.isclose(metadata["radius_km"], RADIUS_KM):
        raise ValueError(
            f"{filename} radius is {metadata['radius_km']:g} km; "
            f"expected {RADIUS_KM:g} km."
        )

    full_rao_count = metadata["seconds"] * 1000 // metadata["trao_ms"]
    expected_rao_indices = np.arange(
        0,
        full_rao_count,
        metadata["rao_step"],
        dtype=int,
    )
    if not np.array_equal(metadata["rao_indices"], expected_rao_indices):
        raise ValueError(
            f"{filename} does not use the regular RAO sampling required by "
            "compute_top3_group_ps_table."
        )
    return metadata


def load_base_pool(filename):
    with Path(filename).open("r", encoding="utf-8") as file:
        records = json.load(file)
    satellite_ids = [int(record["norad_id"]) for record in records]
    if len(satellite_ids) != len(set(satellite_ids)):
        raise ValueError(f"{filename} contains duplicate NORAD IDs.")
    return satellite_ids


def get_plane_member_ids(satellites, representative):
    target_inclination, target_raan = satellite_orbit_values(representative)
    member_ids = set()
    for satellite in satellites:
        inclination, raan = satellite_orbit_values(satellite)
        if abs(inclination - target_inclination) >= (
            INCLINATION_TOLERANCE_DEG
        ):
            continue
        if circular_angle_difference_deg(raan, target_raan) >= (
            RAAN_TOLERANCE_DEG
        ):
            continue
        member_ids.add(int(satellite.model.satnum))
    return member_ids


def build_active_pool(
    all_satellites,
    selected_ids,
    scenario_metadata,
    seconds,
):
    candidate_satellites = [
        satellite
        for satellite in all_satellites
        if int(satellite.model.satnum) in selected_ids
    ]
    timescale = load.timescale()
    midpoint = scenario_metadata["start_dt"] + timedelta(seconds=seconds / 2)
    midpoint_time = timescale.from_datetime(midpoint)
    service_center = wgs84.latlon(CENTER[0], CENTER[1])

    active_satellites = []
    for satellite in candidate_satellites:
        visible, elevation_angle, _ = channel_visibility(
            service_center,
            satellite,
            min_elevation=MIN_ELEVATION_DEG,
            t=midpoint_time,
        )
        if visible and elevation_angle > MIN_ELEVATION_DEG:
            active_satellites.append(satellite)
    return active_satellites


def save_pool_json(satellites, filename):
    records = [
        {
            "fixed_id": fixed_id,
            "name": satellite.name,
            "norad_id": int(satellite.model.satnum),
        }
        for fixed_id, satellite in enumerate(satellites)
    ]
    with Path(filename).open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)


def table_group_summary(filename, plane_member_ids):
    with np.load(filename, allow_pickle=True) as data:
        weight_table = data["group_weight_table"]
        satellite_ids = np.asarray(data["sat_norad_ids"], dtype=int)

    mean_group_counts = {}
    for prefix_length in (1, 2, 3):
        group_counts = [
            len({tuple(group)[:prefix_length] for group in row})
            for row in weight_table
        ]
        mean_group_counts[prefix_length] = float(np.mean(group_counts))

    membership_probabilities = []
    for row in weight_table:
        probability = 0.0
        for group, weight in row.items():
            group_norad_ids = {
                int(satellite_ids[int(index)]) for index in group
            }
            if group_norad_ids.intersection(plane_member_ids):
                probability += float(weight)
        membership_probabilities.append(probability)
    return mean_group_counts, float(np.mean(membership_probabilities))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate only the nested 3-plane and 4-plane Top-3 inputs "
            "required by lab.py Mode 19."
        )
    )
    parser.add_argument("--base-pool", type=Path, default=BASE_POOL)
    parser.add_argument(
        "--template-table",
        type=Path,
        default=TEMPLATE_TABLE,
    )
    parser.add_argument(
        "--plane-3-norad",
        type=int,
        default=PLANE_3_REPRESENTATIVE_NORAD,
    )
    parser.add_argument(
        "--plane-4-norad",
        type=int,
        default=PLANE_4_REPRESENTATIVE_NORAD,
    )
    parser.add_argument("--output-3-pool", type=Path, default=NESTED_3_POOL)
    parser.add_argument("--output-4-pool", type=Path, default=NESTED_4_POOL)
    parser.add_argument(
        "--output-3-table",
        type=Path,
        default=NESTED_3_TABLE,
    )
    parser.add_argument(
        "--output-4-table",
        type=Path,
        default=NESTED_4_TABLE,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    required_inputs = (arguments.base_pool, arguments.template_table)
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing Mode 19 nested-preselection inputs: "
            + ", ".join(missing_inputs)
        )

    output_paths = (
        arguments.output_3_pool,
        arguments.output_4_pool,
        arguments.output_3_table,
        arguments.output_4_table,
    )
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing Mode 19 outputs: "
            + ", ".join(existing_outputs)
        )
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    template = load_template_metadata(arguments.template_table)
    base_ids = load_base_pool(arguments.base_pool)
    if not np.array_equal(
        np.asarray(base_ids, dtype=int),
        template["satellite_ids"],
    ):
        raise ValueError(
            f"{arguments.base_pool} satellite order does not match "
            f"{arguments.template_table}."
        )

    all_satellites = load_starlink_tles(reload=False)
    satellite_by_id = {
        int(satellite.model.satnum): satellite
        for satellite in all_satellites
    }
    required_representatives = (
        arguments.plane_3_norad,
        arguments.plane_4_norad,
    )
    missing_representatives = [
        norad_id
        for norad_id in required_representatives
        if norad_id not in satellite_by_id
    ]
    if missing_representatives:
        raise ValueError(
            "Representative satellites are missing from the current TLE: "
            f"{missing_representatives}"
        )

    scenario_metadata = get_tle_scenario_metadata(all_satellites)
    if scenario_metadata["start_dt_iso"] != template["scenario_start_dt_iso"]:
        raise ValueError(
            f"{arguments.template_table} and the current scenario start "
            "time do not match."
        )
    if scenario_metadata["tle_file_sha256"] != template["tle_file_sha256"]:
        raise ValueError(
            f"{arguments.template_table} and the current TLE file do not match."
        )

    plane_3_representative = satellite_by_id[arguments.plane_3_norad]
    plane_4_representative = satellite_by_id[arguments.plane_4_norad]
    plane_3_member_ids = get_plane_member_ids(
        all_satellites,
        plane_3_representative,
    )
    plane_4_member_ids = get_plane_member_ids(
        all_satellites,
        plane_4_representative,
    )
    if plane_3_member_ids.intersection(plane_4_member_ids):
        raise ValueError("The selected third and fourth orbital planes overlap.")

    nested_3_satellites = build_active_pool(
        all_satellites,
        set(base_ids).union(plane_3_member_ids),
        scenario_metadata,
        template["seconds"],
    )
    nested_4_satellites = build_active_pool(
        all_satellites,
        set(base_ids).union(plane_3_member_ids, plane_4_member_ids),
        scenario_metadata,
        template["seconds"],
    )
    nested_3_ids = {
        int(satellite.model.satnum) for satellite in nested_3_satellites
    }
    nested_4_ids = {
        int(satellite.model.satnum) for satellite in nested_4_satellites
    }
    if not set(base_ids).issubset(nested_3_ids):
        raise ValueError("The generated 3-plane pool does not contain the base pool.")
    if not nested_3_ids.issubset(nested_4_ids):
        raise ValueError("The generated 4-plane pool does not contain the 3-plane pool.")

    np.random.seed(template["random_seed"])
    sample_locations = generate_uniform_locations(
        num_points=template["num_points"],
        center=(template["center_lat"], template["center_lon"]),
        R_km=template["radius_km"],
    )

    with tempfile.TemporaryDirectory(
        prefix=".mode19_nested_",
        dir=Path.cwd(),
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        temporary_3_pool = temporary_directory / arguments.output_3_pool.name
        temporary_4_pool = temporary_directory / arguments.output_4_pool.name
        temporary_3_table = temporary_directory / arguments.output_3_table.name
        temporary_4_table = temporary_directory / arguments.output_4_table.name

        save_pool_json(nested_3_satellites, temporary_3_pool)
        save_pool_json(nested_4_satellites, temporary_4_pool)

        compute_top3_group_ps_table(
            real_sats=nested_3_satellites,
            start_dt=scenario_metadata["start_dt"],
            seconds=template["seconds"],
            trao_ms=template["trao_ms"],
            sample_locations=sample_locations,
            filename=temporary_3_table,
            reference_filename=arguments.template_table,
            orbit_plane_count=3,
            scenario_metadata=scenario_metadata,
            generate_full_table=template["rao_step"] == 1,
            sampled_rao_step=template["rao_step"],
        )
        compute_top3_group_ps_table(
            real_sats=nested_4_satellites,
            start_dt=scenario_metadata["start_dt"],
            seconds=template["seconds"],
            trao_ms=template["trao_ms"],
            sample_locations=sample_locations,
            filename=temporary_4_table,
            reference_filename=arguments.template_table,
            orbit_plane_count=4,
            scenario_metadata=scenario_metadata,
            generate_full_table=template["rao_step"] == 1,
            sampled_rao_step=template["rao_step"],
        )

        summary_3, plane_3_probability = table_group_summary(
            temporary_3_table,
            plane_3_member_ids,
        )
        summary_4, plane_4_probability = table_group_summary(
            temporary_4_table,
            plane_4_member_ids,
        )
        if plane_3_probability <= 0.0:
            raise RuntimeError("The selected third plane never enters a Top-3 group.")
        if plane_4_probability <= 0.0:
            raise RuntimeError("The selected fourth plane never enters a Top-3 group.")

        temporary_3_pool.replace(arguments.output_3_pool)
        temporary_4_pool.replace(arguments.output_4_pool)
        temporary_3_table.replace(arguments.output_3_table)
        temporary_4_table.replace(arguments.output_4_table)

    plane_3_inclination, plane_3_raan = satellite_orbit_values(
        plane_3_representative
    )
    plane_4_inclination, plane_4_raan = satellite_orbit_values(
        plane_4_representative
    )
    print("\n--- Mode 19 Nested Preselection Complete ---")
    print(
        f"Plane 3 representative: NORAD {arguments.plane_3_norad}, "
        f"inclination={plane_3_inclination:.3f} deg, "
        f"RAAN={plane_3_raan:.3f} deg"
    )
    print(
        f"Plane 4 representative: NORAD {arguments.plane_4_norad}, "
        f"inclination={plane_4_inclination:.3f} deg, "
        f"RAAN={plane_4_raan:.3f} deg"
    )
    print(
        f"Satellite-pool sizes: 2-plane={len(base_ids)}, "
        f"3-plane={len(nested_3_satellites)}, "
        f"4-plane={len(nested_4_satellites)}"
    )
    print(
        "3-plane mean groups: "
        + ", ".join(
            f"Top-{prefix}={summary_3[prefix]:.3f}"
            for prefix in (1, 2, 3)
        )
    )
    print(
        "4-plane mean groups: "
        + ", ".join(
            f"Top-{prefix}={summary_4[prefix]:.3f}"
            for prefix in (1, 2, 3)
        )
    )
    print(f"Plane 3 Top-3 membership probability: {plane_3_probability:.6f}")
    print(f"Plane 4 Top-3 membership probability: {plane_4_probability:.6f}")
    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
