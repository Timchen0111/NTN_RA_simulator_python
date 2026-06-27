import json
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
import orbit
from main import channel_visibility
from scenario_time import as_utc_datetime, get_tle_scenario_metadata


def save_satellite_pool(real_sats, filename="fixed_satellite_pool_new.json"):
    records = []

    for fixed_id, sat in enumerate(real_sats):
        records.append({
            "fixed_id": fixed_id,
            "name": sat.name,
            "norad_id": int(sat.model.satnum),
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} satellites to {filename}")


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def estimate_channel_success_probability(elevation_angle, distance_km):
    if elevation_angle <= 0:
        return 0.0

    LOS_PROB = {
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "prob": [0.0, 0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998],
    }
    CHANNEL_PARAMETER = {
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "los_sigma_sf_db":  [1.79, 1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72],
        "nlos_sigma_sf_db": [8.93, 8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52],
        "nlos_cl_db":       [20.87, 19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30],
    }

    elevation_angle = np.clip(elevation_angle, 0, 90)
    p_los = np.interp(elevation_angle, LOS_PROB["elevation_deg"], LOS_PROB["prob"])
    los_sigma = np.interp(elevation_angle, CHANNEL_PARAMETER["elevation_deg"], CHANNEL_PARAMETER["los_sigma_sf_db"])
    nlos_sigma = np.interp(elevation_angle, CHANNEL_PARAMETER["elevation_deg"], CHANNEL_PARAMETER["nlos_sigma_sf_db"])
    nlos_clutter_loss = np.interp(elevation_angle, CHANNEL_PARAMETER["elevation_deg"], CHANNEL_PARAMETER["nlos_cl_db"])

    UE_TX_EIRP_DBM = 23.01
    SAT_RX_GAIN_DBI = 24.0
    FC_GHZ = 2.0
    BANDWIDTH_HZ = 0.4e6
    NOISE_FIGURE_DB = 5.0
    SNR_THRESHOLD_DB = 0.0

    fspl_db = 92.45 + 20 * np.log10(FC_GHZ) + 20 * np.log10(distance_km)
    noise_dbm = -174 + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
    base_margin_db = UE_TX_EIRP_DBM + SAT_RX_GAIN_DBI - fspl_db - noise_dbm - SNR_THRESHOLD_DB

    # Success happens when shadow fading is lower than the remaining SNR margin.
    p_success_los = _normal_cdf(base_margin_db / los_sigma)
    p_success_nlos = _normal_cdf((base_margin_db - nlos_clutter_loss) / nlos_sigma)
    return float(p_los * p_success_los + (1.0 - p_los) * p_success_nlos)

def select_active_satellite_pool(
    real_sats,
    geo,
    start_dt,
    seconds,
    min_elevation=10
):
    """
    Match main.py's initial active satellite filtering before precomputation.
    Only satellites visible from the reference ground point at the scenario
    midpoint are kept.
    """
    ts = load.timescale()
    mid_dt = start_dt + timedelta(seconds=seconds / 2)
    mid_t = ts.from_datetime(mid_dt)

    active_sat_pool = []
    for sat in real_sats:
        visible, elevation_angle, _ = channel_visibility(
            geo,
            sat,
            min_elevation=min_elevation,
            t=mid_t
        )
        if visible and elevation_angle > min_elevation:
            active_sat_pool.append(sat)

    print(
        f"Active Sat Pool Size: {len(active_sat_pool)} "
        f"(from {len(real_sats)} candidates)"
    )
    return active_sat_pool

def generate_uniform_locations(num_points=10000, center=(25.03, 121.56), R_km=200.0):
    """
    產生與 main.py 相同分布的 UE observation points。
    在半徑 R_km 的圓形區域內均勻撒點。
    """
    c_lat, c_lon = center

    lat_bound = R_km / 111.0
    lon_bound = R_km / 100.0

    locations = []

    for _ in range(num_points):
        r = np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)

        lat = c_lat + (r * np.sin(theta)) * lat_bound
        lon = c_lon + (r * np.cos(theta)) * lon_bound

        locations.append(wgs84.latlon(lat, lon))

    return locations

def compute_group_ps_table(
    real_sats,
    start_dt,
    seconds,
    trao_ms,
    sample_locations,
    filename="group_ps_table.npz",
    scenario_metadata=None
):
    if scenario_metadata is None:
        scenario_metadata = get_tle_scenario_metadata()
    if as_utc_datetime(start_dt) != scenario_metadata["start_dt"]:
        raise ValueError(
            "Preselection start_dt must come from the current TLE scenario metadata."
        )
    ts = load.timescale()

    num_rao = seconds * 1000 // trao_ms
    num_sat = len(real_sats)
    num_points = len(sample_locations)

    group_weight_table = []
    group_ps_table = []
    mode3_visible_random_ps_table = []

    for n in range(num_rao):
        current_dt = start_dt + timedelta(milliseconds=n * trao_ms)
        current_t = ts.from_datetime(current_dt)
        group_count = {}
        group_ps_sum = {}
        mode3_visible_random_ps_sum = 0.0
        for loc in sample_locations:
            angles = np.zeros(num_sat)
            distances = np.zeros(num_sat)
            ps_vector = np.zeros(num_sat)
            for k, sat in enumerate(real_sats):
                difference = sat - loc
                topocentric = difference.at(current_t)
                alt, az, distance = topocentric.altaz()
                angle = alt.degrees
                dist_km = distance.km
                angles[k] = angle
                distances[k] = dist_km
                ps_vector[k] = estimate_channel_success_probability(angle, dist_km)
            visible_mask = angles > 10 #以後統一規定10度以上才算visible
            if np.any(visible_mask):
                # Mode 3 baseline: uniform random selection over UE-visible satellites.
                mode3_visible_random_ps_sum += float(np.mean(ps_vector[visible_mask]))
            top2 = np.argsort(angles)[::-1][:2]
            group = (int(top2[0]), int(top2[1]))
            if group not in group_count:
                group_count[group] = 0
                group_ps_sum[group] = np.zeros(num_sat)
            group_count[group] += 1
            group_ps_sum[group] += ps_vector

        weights = {}
        ps_by_group = {}

        for group in group_count:
            weights[group] = group_count[group] / num_points
            ps_by_group[group] = group_ps_sum[group] / group_count[group]

        group_weight_table.append(weights)
        group_ps_table.append(ps_by_group)
        mode3_visible_random_ps_table.append(mode3_visible_random_ps_sum / num_points)

        if n % 10 == 0:
            print(f"RAO {n}/{num_rao}: groups = {len(weights)}")

    np.savez_compressed(
        filename,
        group_weight_table=np.array(group_weight_table, dtype=object),
        group_ps_table=np.array(group_ps_table, dtype=object),
        mode3_visible_random_ps_table=np.array(mode3_visible_random_ps_table, dtype=float),
        sat_norad_ids=np.array([int(sat.model.satnum) for sat in real_sats]),
        scenario_start_dt_iso=scenario_metadata["start_dt_iso"],
        tle_epoch_min_iso=scenario_metadata["tle_epoch_min_iso"],
        tle_epoch_max_iso=scenario_metadata["tle_epoch_max_iso"],
        tle_epoch_median_iso=scenario_metadata["tle_epoch_median_iso"],
        tle_file_sha256=scenario_metadata["tle_file_sha256"],
        seconds=seconds,
        trao_ms=trao_ms,
        num_points=num_points
    )

    print(f"Saved group p_s table to {filename}")

def main(NUM_SAT):
    np.random.seed(42)

    ts = load.timescale()
    scenario_metadata = get_tle_scenario_metadata()
    start_dt = scenario_metadata["start_dt"]
    print(f"Scenario start time from TLE median epoch: {scenario_metadata['start_dt_iso']}")

    t_start = ts.from_datetime(start_dt)

    geo = wgs84.latlon(25.03, 121.56)

    real_sats = orbit.get_relevant_rail_planes(
        t_start,
        geo,
        top_n=NUM_SAT
    )

    # 這裡要有一個初始篩選衛星的過程，挑選會過境的衛星，之前的main是有這個邏輯的。

    real_sats = select_active_satellite_pool(
        real_sats=real_sats,
        geo=geo,
        start_dt=start_dt,
        seconds=200,
        min_elevation=10
    )

    save_satellite_pool(real_sats)

    sample_locations = generate_uniform_locations(
        num_points=1000,
        center=(25.03, 121.56),
        R_km=200.0
    )

    compute_group_ps_table(
        real_sats=real_sats,
        start_dt=start_dt,
        seconds=200,
        trao_ms=100,
        sample_locations=sample_locations,
        filename="group_ps_table.npz",
        scenario_metadata=scenario_metadata
    )

if __name__ == "__main__":
    main(NUM_SAT=4)
