import os
import math
import subprocess

def _gz_publish(topic: str, value_str: str) -> bool:
    cmd = [
        "gz", "topic",
        "-t", topic,
        "-m", "gz.msgs.Double",
        "-p", f"data: {value_str}",
    ]
    print(f"Publishing to {topic}  data={value_str}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("OK")
        return True
    print(f"FAILED\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return False


def lock_servo(servo: int, angle_rad: float, model: str = "standard_vtol_1") -> bool:
    """Lock servo at a fixed angle (rad). Topic: /<model>/aileron_lock/servo_<N>"""
    topic = f"/{model}/aileron_lock/servo_{servo}"
    return _gz_publish(topic, str(angle_rad))


def unlock_servo(servo: int, model: str = "standard_vtol_1") -> bool:
    """Unlock servo by sending nan. Topic: /<model>/aileron_lock/servo_<N>"""
    topic = f"/{model}/aileron_lock/servo_{servo}"
    return _gz_publish(topic, "nan")


def set_servo_limit(servo: int, limit_rad: float, model: str = "standard_vtol_1") -> bool:
    """Set symmetric joint limits ±limit_rad. Topic: /<model>/servo_<N>_limit"""
    topic = f"/{model}/servo_{servo}_limit"
    return _gz_publish(topic, str(limit_rad))


def quaternion_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]

def calc_bank_for_heading(
    current_heading_deg: float,
    desired_heading_deg: float,
    airspeed_ms: float,
    max_bank_deg: float = 30.0,
    g: float = 9.81,
) -> tuple:
    """
    Calculate the bank angle needed to turn from current to desired heading.

    Returns:
        bank_deg     — bank angle to command (positive = right, negative = left)
        turn_rate    — deg/s at that bank angle
        time_to_turn — seconds to complete the turn
    """

    # ── Heading error ────────────────────────────────────────────
    # Normalize to -180 to +180
    heading_error = desired_heading_deg - current_heading_deg
    while heading_error >  180.0:
        heading_error -= 360.0
    while heading_error < -180.0:
        heading_error += 360.0

    # Direction of turn
    turn_direction = 1.0 if heading_error >= 0 else -1.0

    # ── Turn rate needed ─────────────────────────────────────────
    # We want to complete the turn in a reasonable time
    # Use standard rate turn = 3 deg/s as target
    target_turn_rate = 20.0   # deg/s standard rate turn

    # ── Bank angle for standard rate turn ────────────────────────
    # bank = atan(turn_rate_rad * V / g)
    turn_rate_rad = math.radians(target_turn_rate)
    bank_rad      = math.atan(turn_rate_rad * airspeed_ms / g)
    bank_deg      = math.degrees(bank_rad)

    # Clamp to max bank
    bank_deg = min(bank_deg, max_bank_deg)

    # Recompute actual turn rate from clamped bank
    actual_turn_rate = math.degrees(
        g * math.tan(math.radians(bank_deg)) / airspeed_ms
    )

    # Apply direction
    bank_deg      = bank_deg * turn_direction
    actual_turn_rate = actual_turn_rate * turn_direction

    # ── Time to complete turn ────────────────────────────────────
    time_to_turn = abs(heading_error) / abs(actual_turn_rate)

    return bank_deg, actual_turn_rate, time_to_turn