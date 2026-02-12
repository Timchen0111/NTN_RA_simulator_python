import numpy as np
from skyfield.api import load, wgs84
import os

# 增加 top_n 參數，預設為 4
def get_relevant_rail_planes(start_time, location_latlon, tle_url=None, top_n=4):
    """
    Finds orbital planes passing over a specific location and returns all satellites in those planes.
    Only returns the Top-N planes with the most visible satellites.
    """
    
    # 1. Load TLE (Force reload to fix "0 satellites" issue)
    if tle_url is None:
        tle_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    
    print("[Step 1] Downloading TLE data...")
    
    # reload=True ensures we don't use a broken cached file
    try:
        satellites = load.tle_file(tle_url, reload=True)
    except Exception as e:
        print(f"[Error] Failed to download TLE: {e}")
        return []

    print(f"DEBUG: Raw TLE records loaded: {len(satellites)}")
    
    # Debug: Check the first satellite name to ensure filtering logic is correct
    if len(satellites) > 0:
        print(f"DEBUG: First satellite name in DB: '{satellites[0].name}'")

    # Filter for Starlink only
    starlinks = [s for s in satellites if 'STARLINK' in s.name]
    print(f"Total Starlink satellites found: {len(starlinks)}")
    
    if len(starlinks) == 0:
        print("[Error] No Starlink satellites found! Check your internet or TLE source.")
        return []

    # 2. Identify "Seed Satellites" (Visible now)
    print(f"[Step 2] Searching for visible planes at {start_time.utc_strftime('%Y-%m-%d %H:%M:%S')}...")
    
    visible_planes_fingerprint = set() 

    for sat in starlinks:
        try:
            # Calculate Elevation
            difference = sat - location_latlon
            alt, _, _ = difference.at(start_time).altaz()
            
            if alt.degrees > 10: 
                inc = sat.model.inclo
                raan = sat.model.nodeo
                visible_planes_fingerprint.add((inc, raan, sat.name))
        except:
            continue
            
    print(f"Found {len(visible_planes_fingerprint)} seed satellites (visible now).")

    # 3. Expand: Find all members of these planes
    TOLERANCE_RAAN = np.deg2rad(5.0)  
    TOLERANCE_INC = np.deg2rad(1.0)   
    
    final_sats = []
    
    # --- [修改開始] Group unique planes with COUNTS ---
    # 改用 list 存 [inc, raan, count] 以便排序
    plane_candidates = [] 

    for p in visible_planes_fingerprint:
        inc_p, raan_p, name = p
        
        match_found = False
        # 檢查這個種子是否屬於已知的候選面
        for i in range(len(plane_candidates)):
            u_inc, u_raan, count = plane_candidates[i]
            
            if abs(inc_p - u_inc) < TOLERANCE_INC and \
               abs(raan_p - u_raan) < TOLERANCE_RAAN:
                # 找到同一個面，計數 +1
                plane_candidates[i][2] += 1 
                match_found = True
                break
        
        # 如果是新的面，加入列表，初始計數為 1
        if not match_found:
            plane_candidates.append([inc_p, raan_p, 1])

    # 排序：根據 count (index 2) 由大到小排序
    plane_candidates.sort(key=lambda x: x[2], reverse=True)

    # 切片：只取前 Top N，並還原成原本的 (inc, raan) 格式以便下方代碼繼續使用
    unique_planes = [(x[0], x[1]) for x in plane_candidates[:top_n]]
    
    print(f"[Step 3] Selected Top {len(unique_planes)} planes out of {len(plane_candidates)} candidates.")
    # --- [修改結束] ---

    print(f"DEBUG: Retrieving members for selected planes...")

    for sat in starlinks:
        sat_inc = sat.model.inclo
        sat_raan = sat.model.nodeo
        
        match = False
        for (target_inc, target_raan) in unique_planes:
            if abs(sat_inc - target_inc) > TOLERANCE_INC:
                continue
            
            diff_raan = abs(sat_raan - target_raan)
            if diff_raan > np.pi: 
                diff_raan = 2*np.pi - diff_raan
                
            if diff_raan < TOLERANCE_RAAN:
                match = True
                break
        
        if match:
            final_sats.append(sat)

    print(f"[OK] Filtering complete! Retrieved {len(final_sats)} satellites (across {len(unique_planes)} planes).")
    return final_sats

# --- Test Block ---
if __name__ == "__main__":
    from datetime import datetime, timezone
    
    taipei = wgs84.latlon(25.03, 121.56)
    ts = load.timescale()
    
    # Ensure this matches your golden window time
    t = ts.from_datetime(datetime(2026, 2, 12, 20, 42, 0, tzinfo=timezone.utc)) 
    
    # 呼叫時可以指定 top_n，預設為 4
    my_sats = get_relevant_rail_planes(t, taipei, top_n=4)
    
    if len(my_sats) > 0:
        print("Sample Satellites:", [s.name for s in my_sats[:5]])
    else:
        print("No satellites returned.")