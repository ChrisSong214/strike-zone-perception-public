import numpy as np
import pandas as pd

def get_time_at_distance(y_target, y0, vy0, ay):
    """
    Solves for t such that y(t) = y_target.
    y(t) = y0 + vy0*t + 0.5*ay*t^2
    Returns the smallest positive t (or closest to 0 if integrating backwards).
    """
    A = 0.5 * ay
    if A == 0.0:
        raise ZeroDivisionError("Acceleration along y-axis is zero (linear motion)")
    B = vy0
    C = y0 - y_target
    
    delta = B**2 - 4*A*C
    if delta < 0:
        return None
    
    sqrt_delta = np.sqrt(delta)
    t1 = (-B - sqrt_delta) / (2*A)
    t2 = (-B + sqrt_delta) / (2*A)
    
    # We generally want the smallest positive time if we are moving forward
    # But if we are integrating BACKWARDS (e.g. 50 -> 54), we might want negative t.
    # Logic:
    # If y_target < y0 (e.g. 0 < 50), we expect t > 0
    # If y_target > y0 (e.g. 54 > 50), we expect t < 0
    
    # Let's just return both and let caller decide, or robustly pick based on direction
    return t1, t2

def calculate_trajectory(row, time_step=0.005):
    """
    Reconstructs the full 3D trajectory for a pitch row.
    
    Statcast parameters (vx0, vy0, vz0, ax, ay, az) are defined at y=50.
    release_pos_x/y/z are the physical release point (approx y=54).
    plate_x/z are at y=1.417 (front of plate) or y=0 (back of plate). 
    Statcast plate_x/z is usually at y=1.417 (front edge).
    
    Returns: A DataFrame of (t, x, y, z) points.
    """
    # Extract parameters
    # Default to specific values if columns missed (robustness)
    try:
        vx0, vy0, vz0 = row['vx0'], row['vy0'], row['vz0']
        ax, ay, az = row['ax'], row['ay'], row['az']
        # Determine release point Y
        y_release = row.get('release_pos_y')
        if pd.isna(y_release):
            # Fallback: Calculate from release_extension if available
            # Pro doc: release_extension is feet from rubber (60.5)
            ext = row.get('release_extension')
            if pd.notna(ext):
                y_release = 60.5 - ext
            else:
                y_release = 54.5 # Default fallback
        y_plate = 1.417 # Front of plate
        
        # Initial state at y=50 (Statcast reference)
        y_ref = 50.0
        # If the row has release_pos_x/z, we can use them to validate, 
        # but strictly speaking (x,z) at y=50 are what mathematically match the velocity/accel params.
        # We can calculate x_ref, z_ref at y=50 using the equations if we knew x_release.
        # But actually, the params 'vx0' etc are designed to be used with the equations directly.
        # We need the position at y=50. 
        # HOWEVER, Statcast DOES NOT provide x0, z0 at y=50 explicitly in standard CSVs usually?
        # Actually, it provides release_pos_x (at release) and plate_x (at plate).
        # We need to find the constant integration constants.
        
        # Standard Statcast approach:
        # The equation is valid for t=0 at y=50 (where v=v0), but we need x(0), y(0)=50, z(0).
        # We can back-solve x(0), z(0) from release_pos and the time it takes to get from release to 50.
        
        # 1. Find time from y=50 to y=release
        ts = get_time_at_distance(y_release, y_ref, vy0, ay)
        # We expect a negative time (since we are going BACK to release)
        if ts is None: return None
        t_release = min(ts, key=abs) # The one closer to 0
        
        # 2. Back-solve for x_ref, z_ref at t=0 (y=50)
        x_release = row.get('release_pos_x')
        z_release = row.get('release_pos_z')
        if pd.isna(x_release) or pd.isna(z_release):
            return None
        if not np.isfinite(x_release) or not np.isfinite(z_release):
            return None
        
        x_ref = x_release - (vx0 * t_release + 0.5 * ax * t_release**2)
        z_ref = z_release - (vz0 * t_release + 0.5 * az * t_release**2)
        if not np.isfinite(x_ref) or not np.isfinite(z_ref):
            return None
        
        # 3. Find time from y=50 to y=plate
        ts_plate = get_time_at_distance(y_plate, y_ref, vy0, ay)
        if ts_plate is None: return None
        
        # FIX: Pick the smallest positive time (forward to plate)
        # NOT max(), which picks the return trip (e.g. 8s instead of 0.4s)
        valid_times = [t for t in ts_plate if t > 0]
        if not valid_times: return None
        t_final = min(valid_times)
        
        # 4. Generate time steps from t_release to t_final
        times = np.arange(t_release, t_final, time_step)
        if times[-1] < t_final:
            times = np.append(times, t_final)
            
        # 5. Compute position arrays
        xs = x_ref + vx0 * times + 0.5 * ax * times**2
        ys = y_ref + vy0 * times + 0.5 * ay * times**2
        zs = z_ref + vz0 * times + 0.5 * az * times**2
        
        # 6. Compute velocity arrays (v = v0 + a*t, linear under constant accel)
        vxs = vx0 + ax * times
        vys = vy0 + ay * times
        vzs = vz0 + az * times
        
        if not np.isfinite(xs).all() or not np.isfinite(ys).all() or not np.isfinite(zs).all():
            return None
        
        traj_df = pd.DataFrame({
            't': times,
            'x': xs,
            'y': ys,
            'z': zs,
            'vx': vxs,
            'vy': vys,
            'vz': vzs,
            'ax': np.full_like(times, ax),
            'ay': np.full_like(times, ay),
            'az': np.full_like(times, az)
        })
        
        return traj_df

    except Exception as e:
        # print(f"Error calculating trajectory: {e}")
        return None
