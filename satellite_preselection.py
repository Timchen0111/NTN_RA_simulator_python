import json
import numpy as np
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
import orbit
from main import channel_calculator, channel_visibility


def save_satellite_pool(real_sats, filename="fixed_satellite_pool.json"):
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


def estimate_channel_success_probability(elevation_angle, distance_km, mc_trials=200):
    success = 0
    for _ in range(mc_trials):
        if channel_calculator(elevation_angle, distance_km):
            success += 1
    return success / mc_trials

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
    filename="group_ps_table.npz"
):
    ts = load.timescale()

    num_rao = seconds * 1000 // trao_ms
    num_sat = len(real_sats)
    num_points = len(sample_locations)

    group_weight_table = []
    group_ps_table = []

    for n in range(num_rao):
        current_dt = start_dt + timedelta(milliseconds=n * trao_ms)
        current_t = ts.from_datetime(current_dt)
        group_count = {}
        group_ps_sum = {}
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
                if angle > 0:
                    # 單一點不用大量 MC，直接取一次 channel trial
                    ps_vector[k] = 1.0 if channel_calculator(angle, dist_km) else 0.0
                else:
                    ps_vector[k] = 0.0
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

        if n % 10 == 0:
            print(f"RAO {n}/{num_rao}: groups = {len(weights)}")

    np.savez_compressed(
        filename,
        group_weight_table=np.array(group_weight_table, dtype=object),
        group_ps_table=np.array(group_ps_table, dtype=object),
        sat_norad_ids=np.array([int(sat.model.satnum) for sat in real_sats]),
        seconds=seconds,
        trao_ms=trao_ms,
        num_points=num_points
    )

    print(f"Saved group p_s table to {filename}")

def main(NUM_SAT):
    np.random.seed(42)

    ts = load.timescale()

    start_dt = datetime(
        2026, 2, 12, 20, 42, 0,
        tzinfo=timezone.utc
    )

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
        seconds=10,
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
        seconds=10,
        trao_ms=100,
        sample_locations=sample_locations,
        filename="group_ps_table.npz"
    )

if __name__ == "__main__":
    main(NUM_SAT=3)
