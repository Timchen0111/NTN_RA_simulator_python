import json
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
from skyfield.api import load, wgs84

from satellite_preselection import compute_group_ps_table, generate_uniform_locations
from scenario_time import get_tle_scenario_metadata, load_starlink_tles


POOL_DEFINITIONS = (
    ("set_1", (1, 4, 7)),
    ("set_2", (2, 5, 8)),
    ("set_3", (3, 6, 9)),
)
ORBIT_PLANE_COUNT = 3
OUTPUT_TEMPLATE = "fixed_satellite_pool_planes_3_{name}.json"
GROUP_TABLE_TEMPLATE = "group_ps_table_planes_3_{name}.npz"
MIN_ELEVATION_DEG = 10.0
SIMULATION_SECONDS = 200
TRAO_MS = 100
SAMPLE_LOCATION_COUNT = 1000
SERVICE_RADIUS_KM = 200.0
LOCATION_LAT_LON = (25.03, 121.56)
TOLERANCE_RAAN = np.deg2rad(5.0)
TOLERANCE_INC = np.deg2rad(1.0)


def build_ranked_planes(starlinks, start_time, location):
    visible_plane_fingerprints = set()
    visibility_errors = 0

    for satellite in starlinks:
        try:
            altitude, _, _ = (satellite - location).at(start_time).altaz()
            if altitude.degrees > MIN_ELEVATION_DEG:
                visible_plane_fingerprints.add((
                    satellite.model.inclo,
                    satellite.model.nodeo,
                    satellite.name,
                ))
        except Exception:
            visibility_errors += 1

    plane_candidates = []
    for inclination, raan, _ in sorted(visible_plane_fingerprints):
        for plane in plane_candidates:
            if (
                abs(inclination - plane[0]) < TOLERANCE_INC
                and abs(raan - plane[1]) < TOLERANCE_RAAN
            ):
                plane[2] += 1
                break
        else:
            plane_candidates.append([inclination, raan, 1])

    plane_candidates.sort(key=lambda plane: plane[2], reverse=True)
    return plane_candidates, len(visible_plane_fingerprints), visibility_errors


def satellites_in_planes(starlinks, selected_planes):
    selected_satellites = []
    for satellite in starlinks:
        satellite_inclination = satellite.model.inclo
        satellite_raan = satellite.model.nodeo

        for _, target_plane in selected_planes:
            if abs(satellite_inclination - target_plane[0]) > TOLERANCE_INC:
                continue

            raan_difference = abs(satellite_raan - target_plane[1])
            if raan_difference > np.pi:
                raan_difference = 2 * np.pi - raan_difference
            if raan_difference < TOLERANCE_RAAN:
                selected_satellites.append(satellite)
                break

    return selected_satellites


def filter_midpoint_visible(satellites, location, midpoint_time):
    active_satellites = []
    for satellite in satellites:
        altitude, _, _ = (satellite - location).at(midpoint_time).altaz()
        if altitude.degrees > MIN_ELEVATION_DEG:
            active_satellites.append(satellite)
    return active_satellites


def write_pool_exclusively(satellites, output_path):
    records = [
        {
            "fixed_id": fixed_id,
            "name": satellite.name,
            "norad_id": int(satellite.model.satnum),
        }
        for fixed_id, satellite in enumerate(satellites)
    ]
    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2, ensure_ascii=False)


def print_pool_summary(summary):
    print(f"\n{summary['name']}  ranks={summary['ranks']}")
    for plane in summary["planes"]:
        print(
            f"  rank {plane['rank']}: "
            f"inclination={plane['inclination_deg']:.3f} deg, "
            f"RAAN={plane['raan_deg']:.3f} deg, "
            f"visible satellites={plane['visible_satellite_count']}"
        )
    print(
        "  Satellites in the three complete planes: "
        f"{summary['plane_satellite_count']}"
    )
    print(f"  Final midpoint-visible satellites: {summary['final_satellite_count']}")
    print(f"  Satellite pool: {summary['output_path']}")
    print(
        f"  Group p_s table: {summary['group_table_path']} "
        f"({summary['group_table_size_bytes']:,} bytes, "
        f"{summary['group_table_elapsed_seconds']:.1f} s)"
    )
    print(
        "  NORAD IDs: "
        + ", ".join(str(norad_id) for norad_id in summary["norad_ids"])
    )


def main():
    started_at = perf_counter()
    project_directory = Path(__file__).resolve().parent
    selected_ranks = [
        rank
        for _, ranks in POOL_DEFINITIONS
        for rank in ranks
    ]
    if len(selected_ranks) != len(set(selected_ranks)):
        raise ValueError("POOL_DEFINITIONS contains overlapping plane ranks.")
    output_paths = {
        name: project_directory / OUTPUT_TEMPLATE.format(name=name)
        for name, _ in POOL_DEFINITIONS
    }
    group_table_paths = {
        name: project_directory / GROUP_TABLE_TEMPLATE.format(name=name)
        for name, _ in POOL_DEFINITIONS
    }

    existing_outputs = [
        path
        for path in (*output_paths.values(), *group_table_paths.values())
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing satellite pools: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    starlinks = load_starlink_tles()
    scenario_metadata = get_tle_scenario_metadata(starlinks)
    timescale = load.timescale()
    start_time = timescale.from_datetime(scenario_metadata["start_dt"])
    midpoint_datetime = scenario_metadata["start_dt"] + timedelta(
        seconds=SIMULATION_SECONDS / 2
    )
    midpoint_time = timescale.from_datetime(midpoint_datetime)
    location = wgs84.latlon(*LOCATION_LAT_LON)

    ranked_planes, visible_count, visibility_errors = build_ranked_planes(
        starlinks,
        start_time,
        location,
    )
    highest_required_rank = max(
        rank
        for _, ranks in POOL_DEFINITIONS
        for rank in ranks
    )
    if len(ranked_planes) < highest_required_rank:
        raise ValueError(
            f"Only {len(ranked_planes)} candidate planes were found; "
            f"rank {highest_required_rank} is required."
        )

    summaries = []
    pending_outputs = []
    for name, ranks in POOL_DEFINITIONS:
        selected_planes = [
            (rank, ranked_planes[rank - 1])
            for rank in ranks
        ]
        plane_satellites = satellites_in_planes(starlinks, selected_planes)
        final_satellites = filter_midpoint_visible(
            plane_satellites,
            location,
            midpoint_time,
        )
        pending_outputs.append({
            "name": name,
            "ranks": ranks,
            "satellites": final_satellites,
            "pool_path": output_paths[name],
            "group_table_path": group_table_paths[name],
        })
        summaries.append({
            "name": name,
            "ranks": ranks,
            "planes": [
                {
                    "rank": rank,
                    "inclination_deg": float(np.rad2deg(plane[0])),
                    "raan_deg": float(np.rad2deg(plane[1]) % 360.0),
                    "visible_satellite_count": int(plane[2]),
                }
                for rank, plane in selected_planes
            ],
            "plane_satellite_count": len(plane_satellites),
            "final_satellite_count": len(final_satellites),
            "norad_ids": [
                int(satellite.model.satnum)
                for satellite in final_satellites
            ],
            "output_path": output_paths[name].name,
            "group_table_path": group_table_paths[name].name,
        })

    for left_index, left in enumerate(pending_outputs):
        left_ids = {
            int(satellite.model.satnum)
            for satellite in left["satellites"]
        }
        for right in pending_outputs[left_index + 1:]:
            right_ids = {
                int(satellite.model.satnum)
                for satellite in right["satellites"]
            }
            overlap = sorted(left_ids.intersection(right_ids))
            if overlap:
                raise ValueError(
                    f"{left['name']} and {right['name']} overlap on NORAD IDs: "
                    + ", ".join(str(norad_id) for norad_id in overlap)
                )

    np.random.seed(42)
    sample_locations = generate_uniform_locations(
        num_points=SAMPLE_LOCATION_COUNT,
        center=LOCATION_LAT_LON,
        R_km=SERVICE_RADIUS_KM,
    )

    for pending, summary in zip(pending_outputs, summaries):
        print(
            f"\nComputing group p_s table for {pending['name']} "
            f"with plane ranks {pending['ranks']}...",
            flush=True,
        )
        table_started_at = perf_counter()
        compute_group_ps_table(
            real_sats=pending["satellites"],
            start_dt=scenario_metadata["start_dt"],
            seconds=SIMULATION_SECONDS,
            trao_ms=TRAO_MS,
            sample_locations=sample_locations,
            filename=str(pending["group_table_path"]),
            scenario_metadata=scenario_metadata,
            extra_metadata={
                "orbit_plane_count": ORBIT_PLANE_COUNT,
                "pool_set": pending["name"],
                "selected_plane_ranks": np.array(pending["ranks"], dtype=int),
                "min_elevation_deg": MIN_ELEVATION_DEG,
                "radius_km": SERVICE_RADIUS_KM,
            },
        )
        summary["group_table_elapsed_seconds"] = (
            perf_counter() - table_started_at
        )
        summary["group_table_size_bytes"] = pending["group_table_path"].stat().st_size

    for pending in pending_outputs:
        write_pool_exclusively(pending["satellites"], pending["pool_path"])

    print("=== Non-overlapping Three-plane Satellite Pools ===")
    print(f"Scenario start (UTC): {scenario_metadata['start_dt_iso']}")
    print(f"TLE SHA-256: {scenario_metadata['tle_file_sha256']}")
    print(f"Starlink satellites loaded: {len(starlinks)}")
    print(f"Visible satellites at start (>10 deg): {visible_count}")
    print(f"Ranked candidate planes: {len(ranked_planes)}")
    print(f"Visibility calculation errors: {visibility_errors}")
    print(
        f"Group table sampling: {SIMULATION_SECONDS} s, {TRAO_MS} ms/RAO, "
        f"{SAMPLE_LOCATION_COUNT} locations, radius {SERVICE_RADIUS_KM:g} km"
    )
    for summary in summaries:
        print_pool_summary(summary)

    print(f"\nCompleted in {perf_counter() - started_at:.2f} seconds.")


if __name__ == "__main__":
    main()
