#!/usr/bin/env python3
"""
inject_actuator_failure.py
Inject servo failures into a Gazebo VTOL model via gz transport topics.

Subcommands:
  lock    -- Lock a servo at a fixed angle (rad)
  unlock  -- Unlock a servo (restores normal control)
  limit   -- Set symmetric joint limits for a servo (±rad)

Examples:
  python inject_actuator_failure.py lock   --servo 0 --angle 0.3
  python inject_actuator_failure.py lock   --servo 1 --angle -0.5
  python inject_actuator_failure.py unlock --servo 0
  python inject_actuator_failure.py unlock --servo 1
  python inject_actuator_failure.py limit  --servo 0 --value 0.2
  python inject_actuator_failure.py limit  --servo 1 --value 0.4
  python inject_actuator_failure.py lock   --servo 0 --angle 0.3 --model standard_vtol_2
"""

import math
import subprocess
import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="Inject actuator failures into a Gazebo VTOL model."
    )
    parser.add_argument(
        "--model", type=str, default="standard_vtol_1",
        help="Gazebo model name (default: standard_vtol_1)."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- lock ----------------------------------------------------------------
    p_lock = subparsers.add_parser("lock", help="Lock a servo at a fixed angle.")
    p_lock.add_argument("--servo", type=int, choices=[0, 1], required=True,
                        help="Servo index (0 or 1).")
    p_lock.add_argument("--angle", type=float, required=True,
                        help="Lock angle in radians.")

    # -- unlock --------------------------------------------------------------
    p_unlock = subparsers.add_parser("unlock", help="Unlock a servo (restore normal control).")
    p_unlock.add_argument("--servo", type=int, choices=[0, 1], required=True,
                          help="Servo index (0 or 1).")

    # -- limit ---------------------------------------------------------------
    p_limit = subparsers.add_parser("limit", help="Set symmetric joint limits ±value rad.")
    p_limit.add_argument("--servo", type=int, choices=[0, 1], required=True,
                         help="Servo index (0 or 1).")
    p_limit.add_argument("--value", type=float, required=True,
                         help="Limit magnitude in radians (applied as ±value).")

    args = parser.parse_args()

    if args.command == "lock":
        lock_servo(args.servo, args.angle, args.model)
    elif args.command == "unlock":
        unlock_servo(args.servo, args.model)
    elif args.command == "limit":
        set_servo_limit(args.servo, args.value, args.model)


if __name__ == "__main__":
    main()
