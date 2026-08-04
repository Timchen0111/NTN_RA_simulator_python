import hashlib
from pathlib import Path

import numpy as np

from main import load_fixed_satellites
from satellite_preselection import (
    compute_group_ps_table,
    generate_uniform_locations,
)
from scenario_time import get_tle_scenario_metadata


REFERENCE_TABLE = Path("group_ps_table.npz")
FIXED_SATELLITE_POOL = Path("fixed_satellite_pool.json")
RANDOM_SEED = 42
CENTER = (25.03, 121.56)
RADIUS_OUTPUTS = (
    (100.0, Path("group_ps_table_radius_100km.npz")),
    (300.0, Path("group_ps_table_radius_300km.npz")),
)


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_configuration(filename=REFERENCE_TABLE):
    required_keys = {
        "sat_norad_ids",
        "scenario_start_dt_iso",
        "tle_file_sha256",
        "seconds",
        "trao_ms",
        "num_points",
    }
    with np.load(filename, allow_pickle=True) as reference:
        missing_keys = required_keys.difference(reference.files)
        if missing_keys:
            raise ValueError(
                f"{filename} is missing required fields: {sorted(missing_keys)}"
            )
        return {
            "sat_norad_ids": np.asarray(reference["sat_norad_ids"], dtype=int),
            "scenario_start_dt_iso": str(reference["scenario_start_dt_iso"]),
            "tle_file_sha256": str(reference["tle_file_sha256"]),
            "seconds": int(reference["seconds"]),
            "trao_ms": int(reference["trao_ms"]),
            "num_points": int(reference["num_points"]),
        }


def validate_reference_configuration(reference, scenario_metadata, real_sats):
    if reference["scenario_start_dt_iso"] != scenario_metadata["start_dt_iso"]:
        raise ValueError(
            "Reference table and current TLE scenario start time do not match."
        )
    if reference["tle_file_sha256"] != scenario_metadata["tle_file_sha256"]:
        raise ValueError("Reference table and current TLE file do not match.")

    actual_satellite_ids = np.array(
        [int(satellite.model.satnum) for satellite in real_sats],
        dtype=int,
    )
    if not np.array_equal(actual_satellite_ids, reference["sat_norad_ids"]):
        raise ValueError(
            "fixed_satellite_pool.json does not match the satellite IDs and order "
            "stored in group_ps_table.npz."
        )


def generate_radius_table(
    radius_km,
    output_filename,
    real_sats,
    reference,
    scenario_metadata,
):
    output_filename = Path(output_filename)
    if output_filename.resolve() == REFERENCE_TABLE.resolve():
        raise ValueError("Refusing to overwrite the existing baseline table.")
    if output_filename.exists():
        raise FileExistsError(
            f"{output_filename} already exists; remove or rename it explicitly "
            "before regenerating this radius table."
        )

    # Resetting the seed for each radius preserves the same normalized UE
    # locations (r and theta); only their physical distance from the center changes.
    np.random.seed(RANDOM_SEED)
    sample_locations = generate_uniform_locations(
        num_points=reference["num_points"],
        center=CENTER,
        R_km=radius_km,
    )

    print(
        f"Generating radius table: radius={radius_km:g} km, "
        f"satellites={len(real_sats)}, seconds={reference['seconds']}, "
        f"RAO={reference['trao_ms']} ms, points={reference['num_points']}"
    )
    compute_group_ps_table(
        real_sats=real_sats,
        start_dt=scenario_metadata["start_dt"],
        seconds=reference["seconds"],
        trao_ms=reference["trao_ms"],
        sample_locations=sample_locations,
        filename=output_filename,
        scenario_metadata=scenario_metadata,
        extra_metadata={
            "radius_km": float(radius_km),
            "center_lat": float(CENTER[0]),
            "center_lon": float(CENTER[1]),
            "random_seed": int(RANDOM_SEED),
            "source_satellite_pool": str(FIXED_SATELLITE_POOL),
            "source_satellite_pool_sha256": sha256_file(FIXED_SATELLITE_POOL),
            "source_reference_table": str(REFERENCE_TABLE),
            "source_reference_table_sha256": sha256_file(REFERENCE_TABLE),
        },
    )


def main():
    if not FIXED_SATELLITE_POOL.exists():
        raise FileNotFoundError(FIXED_SATELLITE_POOL)
    if not REFERENCE_TABLE.exists():
        raise FileNotFoundError(REFERENCE_TABLE)

    reference = load_reference_configuration()
    scenario_metadata = get_tle_scenario_metadata()
    real_sats = load_fixed_satellites(FIXED_SATELLITE_POOL)
    validate_reference_configuration(reference, scenario_metadata, real_sats)

    print(
        f"Using the existing fixed satellite pool: {len(real_sats)} satellites"
    )
    print(f"Satellite pool SHA-256: {sha256_file(FIXED_SATELLITE_POOL)}")
    print(f"Reference table SHA-256: {sha256_file(REFERENCE_TABLE)}")

    for radius_km, output_filename in RADIUS_OUTPUTS:
        generate_radius_table(
            radius_km=radius_km,
            output_filename=output_filename,
            real_sats=real_sats,
            reference=reference,
            scenario_metadata=scenario_metadata,
        )

    print("Completed 100 km and 300 km radius preselection tables.")


if __name__ == "__main__":
    main()
