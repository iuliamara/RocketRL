import numpy as np


def pd_action(observation):
    """Position/velocity feedback with gravity compensation in the toy simulator."""
    x, height, z = observation[:3] * 12.0
    vx, vy, vz = observation[3:6] * 8.0
    up = observation[6:9]
    fuel = observation[12] * 0.4
    mass = 1.0 + fuel + 0.6 * observation[13]
    efficiency = 1.0 + observation[14]
    wind_x, wind_z = observation[15:17]
    target_vy = -min(1.0, 0.42 + 0.28 * max(height, 0))
    fy = max(0.3, mass * (3.2 + 2.2 * (target_vy - vy)))
    fx = mass * (-0.65 * x - 1.7 * vx) - wind_x - fy * up[0]
    fz = mass * (-0.65 * z - 1.7 * vz) - wind_z - fy * up[2]
    gimbal = np.clip(np.array([fx, fz]) / (0.38 * fy), -1.0, 1.0)
    force = fy * np.sqrt(1.0 + np.square(0.38 * gimbal).sum())
    throttle = min(float(force / (10.0 * efficiency)), 1.0)
    return np.array([2.0 * throttle - 1.0, *gimbal], dtype=np.float32)
