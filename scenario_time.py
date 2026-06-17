import hashlib
import os
from datetime import datetime, timezone

import numpy as np
from skyfield.api import load


TLE_FILENAME = "starlink_tle.txt"
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"


def as_utc_datetime(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def skyfield_time_to_utc_datetime(t):
    return as_utc_datetime(t.utc_datetime())


def load_tle_file(tle_url=None, filename=TLE_FILENAME, reload=None):
    if tle_url is None:
        tle_url = TLE_URL
    if reload is None:
        reload = not os.path.exists(filename)
    return load.tle_file(tle_url, filename=filename, reload=reload)


def load_starlink_tles(tle_url=None, filename=TLE_FILENAME, reload=None):
    satellites = load_tle_file(tle_url=tle_url, filename=filename, reload=reload)
    return [sat for sat in satellites if "STARLINK" in sat.name]


def tle_epoch_datetimes(satellites):
    epochs = [skyfield_time_to_utc_datetime(sat.epoch) for sat in satellites]
    if len(epochs) == 0:
        raise ValueError("No TLE epochs found.")
    return epochs


def median_datetime(datetimes):
    timestamps = np.array([dt.timestamp() for dt in datetimes], dtype=float)
    return datetime.fromtimestamp(float(np.median(timestamps)), tz=timezone.utc)


def file_sha256(filename=TLE_FILENAME):
    digest = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_tle_scenario_metadata(satellites=None, filename=TLE_FILENAME):
    if satellites is None:
        satellites = load_starlink_tles(filename=filename)
    epochs = tle_epoch_datetimes(satellites)
    start_dt = median_datetime(epochs)
    return {
        "start_dt": start_dt,
        "start_dt_iso": start_dt.isoformat(),
        "tle_epoch_min_iso": min(epochs).isoformat(),
        "tle_epoch_max_iso": max(epochs).isoformat(),
        "tle_epoch_median_iso": start_dt.isoformat(),
        "tle_file_sha256": file_sha256(filename),
    }
