import math

import numpy as np


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_channel_success_probability(elevation_angle, distance_km):
    if elevation_angle <= 0 or distance_km <= 0:
        return 0.0

    los_prob = {
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "prob": [0.0, 0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998],
    }
    channel_parameter = {
        "elevation_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "los_sigma_sf_db":  [1.79, 1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72],
        "nlos_sigma_sf_db": [8.93, 8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52],
        "nlos_cl_db":       [20.87, 19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30],
    }

    elevation_angle = np.clip(elevation_angle, 0, 90)
    p_los = np.interp(elevation_angle, los_prob["elevation_deg"], los_prob["prob"])
    los_sigma = np.interp(elevation_angle, channel_parameter["elevation_deg"], channel_parameter["los_sigma_sf_db"])
    nlos_sigma = np.interp(elevation_angle, channel_parameter["elevation_deg"], channel_parameter["nlos_sigma_sf_db"])
    nlos_clutter_loss = np.interp(elevation_angle, channel_parameter["elevation_deg"], channel_parameter["nlos_cl_db"])

    ue_tx_eirp_dbm = 23.01
    sat_rx_gain_dbi = 24.0
    fc_ghz = 2.0
    bandwidth_hz = 0.4e6
    noise_figure_db = 5.0
    snr_threshold_db = 0.0

    fspl_db = 92.45 + 20 * np.log10(fc_ghz) + 20 * np.log10(distance_km)
    noise_dbm = -174 + 10 * np.log10(bandwidth_hz) + noise_figure_db
    base_margin_db = ue_tx_eirp_dbm + sat_rx_gain_dbi - fspl_db - noise_dbm - snr_threshold_db

    p_success_los = _normal_cdf(base_margin_db / los_sigma)
    p_success_nlos = _normal_cdf((base_margin_db - nlos_clutter_loss) / nlos_sigma)
    return float(p_los * p_success_los + (1.0 - p_los) * p_success_nlos)
