import numpy as np
import json  # [修改點 1] 匯入 json 模組
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta

# --- 1. 初始化設定 ---
ts = load.timescale()

# TLE 資料
line1 = '1 44714U 19074B   25216.88326253  .00000799  00000+0  72522-4 0  9990'
line2 = '2 44714  53.0549  79.9036 0001271  84.0186 276.0948 15.06396482316077'
satellite = EarthSatellite(line1, line2, 'Starlink 1008', ts)

def get_satellite_position(time):
    """取得衛星在指定時間的地理位置（經緯度、高度）"""
    geocentric = satellite.at(time)
    subpoint = wgs84.subpoint(geocentric)
    latitude = subpoint.latitude.degrees
    longitude = subpoint.longitude.degrees
    altitude = subpoint.elevation.km
    return latitude, longitude, altitude

print(get_satellite_position(ts.now()))  # 範例：取得當前時間的衛星位置