#!/usr/bin/env python3

import math
import time
import collections
import curses
import rclpy
import csv, os
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleCommand,
    VehicleStatus,
    VehicleAttitude,
    VehicleLocalPosition,
    AirspeedValidated,
    OffboardControlMode,
    VehicleAttitudeSetpoint,
)

# =============================================================================
# DAA CONFIG
# =============================================================================
VEHICLE_TYPE_FW = 0
VEHICLE_TYPE_MC = 1
VEHICLE_TYPE_VTOL = 2

LOG_NAME = "Roll_angle_20_SEP_125_Danger_150"   # change this before each run
DAA_ALERT_M      = 200.0
DAA_DANGER_M     = 150.0
SEP_DISTANCE_M   = 80
LOOKAHEAD_S      = 30.0

MAX_TURN_DEG     = 40.0
HEADING_STEPS    = 3600

FALLBACK_STEP_M  = 5.0
FALLBACK_MIN_M   = 5.0

DAA_STRATEGY     = "closest"

TARGET_AIRSPEED   = 20   # m/s — set to your cruise airspeed
TARGET_ALTITUDE   = None   # set at runtime from current altitude
ALTITUDE_GAIN     = 0.02   # pitch correction per metre of altitude error
AIRSPEED_GAIN     = 0.05   # thrust correction per m/s of airspeed error

# =============================================================================
# VEHICLE MODES
# =============================================================================
CMD_SET_MODE         = 176
MAV_MODE_FLAG_CUSTOM = 1
MAIN_MODE_AUTO       = 4
MAIN_MODE_OFFBOARD   = 6
SUB_MODE_AUTO_LOITER = 3



# =============================================================================
# DAA HELPER METHODS
# =============================================================================
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


def is_diverging(obs_pos, obs_vel, intr_pos, intr_vel):
    rel_x = intr_pos[0] - obs_pos[0]
    rel_y = intr_pos[1] - obs_pos[1]
    vrx   = intr_vel[0] - obs_vel[0]
    vry   = intr_vel[1] - obs_vel[1]
    return (rel_x * vrx + rel_y * vry) > 0


def compute_forbidden_cone(obs_pos, intr_pos, sep_dist):
    rel_x = intr_pos[0] - obs_pos[0]
    rel_y = intr_pos[1] - obs_pos[1]
    dist  = math.sqrt(rel_x**2 + rel_y**2)
    cone_center = math.atan2(rel_y, rel_x)
    if dist > sep_dist:
        cone_half = math.asin(min(sep_dist / dist, 1.0))
    else:
        cone_half = math.pi / 2.0
    return cone_center, cone_half, dist


def is_in_forbidden_zone(vr_angle, cone_center, cone_half):
    diff = vr_angle - cone_center
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    return abs(diff) < cone_half


def scan_headings(obs_vel, intr_vel, cone_center, cone_half,
                  current_heading_rad, max_turn_rad, strategy):
    obs_speed = math.sqrt(obs_vel[0]**2 + obs_vel[1]**2)
    if obs_speed < 0.1:
        return None

    step_rad = (2 * max_turn_rad) / HEADING_STEPS
    allowed  = []

    for i in range(HEADING_STEPS + 1):
        candidate    = current_heading_rad - max_turn_rad + i * step_rad
        new_vx       = obs_speed * math.cos(candidate)
        new_vy       = obs_speed * math.sin(candidate)
        new_vrx      = new_vx - intr_vel[0]
        new_vry      = new_vy - intr_vel[1]
        new_vr_angle = math.atan2(new_vry, new_vrx)
        if not is_in_forbidden_zone(new_vr_angle, cone_center, cone_half):
            allowed.append(candidate)

    if not allowed:
        return None

    if strategy == "closest":
        return min(allowed, key=lambda a: abs(a - current_heading_rad))
    else:
        groups, current_group = [], [allowed[0]]
        for j in range(1, len(allowed)):
            if allowed[j] - allowed[j-1] < step_rad * 2:
                current_group.append(allowed[j])
            else:
                groups.append(current_group)
                current_group = [allowed[j]]
        groups.append(current_group)
        largest = max(groups, key=len)
        return largest[len(largest) // 2]


def compute_evasive_heading(obs_pos, obs_vel, intr_pos, intr_vel,
                             strategy="closest"):
    rel_x = intr_pos[0] - obs_pos[0]
    rel_y = intr_pos[1] - obs_pos[1]
    dist  = math.sqrt(rel_x**2 + rel_y**2)

    current_heading_rad = math.atan2(obs_vel[1], obs_vel[0])
    current_heading_deg = math.degrees(current_heading_rad)
    max_turn_rad        = math.radians(MAX_TURN_DEG)

    if is_diverging(obs_pos, obs_vel, intr_pos, intr_vel):
        return None, current_heading_deg, dist, "NORMAL", strategy
    if dist > DAA_ALERT_M:
        return None, current_heading_deg, dist, "NORMAL", strategy
    if dist > DAA_DANGER_M:
        return None, current_heading_deg, dist, "ALERT", strategy

    current_sep = SEP_DISTANCE_M
    while current_sep >= FALLBACK_MIN_M:
        cone_center, cone_half, _ = compute_forbidden_cone(
            obs_pos, intr_pos, current_sep)
        evasive_rad = scan_headings(
            obs_vel, intr_vel, cone_center, cone_half,
            current_heading_rad, max_turn_rad, strategy)
        if evasive_rad is not None:
            evasive_deg = math.degrees(evasive_rad)
            state = "DANGER" if current_sep == SEP_DISTANCE_M \
                else f"DANGER_FALLBACK({current_sep:.0f}m)"
            return evasive_deg, current_heading_deg, dist, state, strategy
        current_sep -= FALLBACK_STEP_M

    return current_heading_deg, current_heading_deg, dist, \
           "DANGER_NO_SOLUTION", strategy

class DAALogger:
    """
    Writes one row per DAA loop tick to a CSV file.
    All distances in metres, angles in degrees, thrust normalised 0-1.
    """

    FIELDS = [
        # ── Time ────────────────────────────────────────────────
        "timestamp_s",          # float  — wall-clock seconds since epoch

        # ── Observer (ownship) ──────────────────────────────────
        "obs_x_m",              # float  — NED x position
        "obs_y_m",              # float  — NED y position
        "obs_vx_ms",            # float  — velocity x
        "obs_vy_ms",            # float  — velocity y
        "obs_alt_m",            # float  — altitude (from -z)
        "obs_airspeed_ms",      # float  — true airspeed

        # ── Intruder ────────────────────────────────────────────
        "intr_x_m",
        "intr_y_m",
        "intr_vx_ms",
        "intr_vy_ms",

        # ── Separation ─────────────────────────────────────────
        "separation_m",         # float  — current horizontal distance

        # ── DAA output ─────────────────────────────────────────
        "daa_state",            # str    — NORMAL / ALERT / DANGER / …
        "current_hdg_deg",      # float  — ownship heading (math convention, deg)
        "evasive_hdg_deg",      # float  — commanded evasive heading (NaN if none)
        "bank_cmd_deg",         # float  — commanded bank angle (NaN if none)
        "heading_error_deg",    # float  — error between current and evasive heading

        # ── Control outputs ────────────────────────────────────
        "thrust_cmd",           # float  — normalised thrust 0-1
        "pitch_cmd_deg",        # float  — commanded pitch angle

        # ── Vehicle state ──────────────────────────────────────
        "nav_state",            # int    — PX4 nav_state enum
        "in_offboard",          # int    — 1 if nav_state == 14, else 0
    ]

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.expanduser("~")
        ts  = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{LOG_NAME}_{ts}.csv")
        self._f  = open(path, "w", newline="")
        self._w  = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._path = path
        print(f"[DAALogger] Logging to {path}")

    def write(self, row: dict):
        """Write one row; missing keys default to empty string."""
        self._w.writerow({f: row.get(f, "") for f in self.FIELDS})

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()
        print(f"[DAALogger] Closed {self._path}")

class RTANode(Node):
    def __init__(self):
        super().__init__("rta_system")

        self.nav_state         = 0
        self.state             = "IDLE"
        self.state_timer       = 0.0
        self.current_roll      = 0.0
        self.current_pitch     = 0.0
        self.current_yaw       = 0.0
        self.current_altitude  = 0.0
        self.current_airspeed  = 0.0
        self.attitude_received = False
        self.locked_yaw        = 0.0
        self.locked_pitch      = 0.0
        self.target_altitude   = 0.0
        self.base_thrust       = 0.6   # starting thrust

        # ── State ─────────────────────────────────────────────────
        self._obs_pos      = None
        self._obs_vel      = None
        self._intr_pos     = None
        self._intr_vel     = None
        self._last_daa     = None
        self._strategy     = DAA_STRATEGY
        self._vehicle_type = VEHICLE_TYPE_FW
        self._scr          = None
        self._event_log    = collections.deque(maxlen=8)
        self._daa_logger = DAALogger()


        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, 
            f"/px4_1/fmu/in/vehicle_command", qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode, 
            f"/px4_1/fmu/in/offboard_control_mode", qos)
        self.attitude_pub = self.create_publisher(VehicleAttitudeSetpoint,
            f"/px4_1/fmu/in/vehicle_attitude_setpoint_v1", qos)

        # Subscribers
        self.create_subscription(VehicleStatus, 
            f"/px4_1/fmu/out/vehicle_status_v4", self._status_cb, qos)
        self.create_subscription(VehicleAttitude, 
            f"/px4_1/fmu/out/vehicle_attitude", self._attitude_cb, qos)
        self.create_subscription(AirspeedValidated, 
            f"/px4_1/fmu/out/airspeed_validated_v1", self._airspeed_cb, qos)
        self.create_subscription(VehicleLocalPosition,
            "/px4_1/fmu/out/vehicle_local_position_v1", self._obs_pos_cb, qos)
        self.create_subscription(VehicleLocalPosition,
            "/px4_2/fmu/out/vehicle_local_position_v1", self._intr_pos_cb, qos)

        #self.create_timer(0.01, self._tick)
        #self.get_logger().info("FwOverride ready. Waiting for data...")

        self.get_logger().info("Node started. Waiting 3s for topics...")
        time.sleep(3.0)

        self.create_timer(0.1, self._daa_loop)
        self.create_timer(0.5, self._debug_print)

    # ── Helpers ───────────────────────────────────────────────────
    def _log(self, msg: str):
        self._event_log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    # ── Callbacks ─────────────────────────────────────────────────
    def _obs_pos_cb(self, msg: VehicleLocalPosition):
        self._obs_pos = (msg.x, msg.y)
        self._obs_vel = (msg.vx, msg.vy)

        # NED frame — z is negative altitude
        self.current_altitude = -msg.z

    def _intr_pos_cb(self, msg: VehicleLocalPosition):
        self._intr_pos = (msg.x, msg.y)
        self._intr_vel = (msg.vx, msg.vy)

    def _vehicle_status_cb(self, msg: VehicleStatus):
        prev = self._vehicle_type
        self._vehicle_type = msg.vehicle_type
        if msg.vehicle_type != prev:
            vtol_map = {VEHICLE_TYPE_FW: "FW", VEHICLE_TYPE_MC: "MC",
                        VEHICLE_TYPE_VTOL: "TRANS"}
            self._log(f"Vehicle: {vtol_map.get(prev,'?')} -> "
                      f"{vtol_map.get(msg.vehicle_type,'?')}")

    def _status_cb(self, msg: VehicleStatus):
        self.nav_state = msg.nav_state

    def _attitude_cb(self, msg: VehicleAttitude):
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        self.current_roll, self.current_pitch, self.current_yaw = quaternion_to_euler(q)
        if not self.attitude_received:
            self.attitude_received = True
            self.get_logger().info("Attitude received ✓")

    def _airspeed_cb(self, msg: AirspeedValidated):
        self.current_airspeed = msg.true_airspeed_m_s

    # ── Helpers ──────────────────────────────────────────────────

    def _send_cmd(self, command, p1=0.0, p2=0.0, p3=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(p1)
        msg.param2           = float(p2)
        msg.param3           = float(p3)
        msg.target_system    = 2
        msg.target_component = 1
        msg.source_system    = 255
        msg.source_component = 0
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.timestamp         = int(self.get_clock().now().nanoseconds / 1000)
        msg.position          = False
        msg.velocity          = False
        msg.acceleration      = False
        msg.attitude          = True
        msg.body_rate         = False
        msg.thrust_and_torque = False
        msg.direct_actuator   = False
        self.offboard_pub.publish(msg)

    def _compute_thrust(self):
        """Adjust thrust to maintain target airspeed."""
        airspeed_error = TARGET_AIRSPEED - self.current_airspeed
        thrust = self.base_thrust + (AIRSPEED_GAIN * airspeed_error)
        # Clamp between 0.01 and 0.9
        return max(0.01, min(0.9, thrust))

    def _compute_pitch(self):
        """Adjust pitch to maintain target altitude."""
        altitude_error = self.target_altitude - self.current_altitude
        pitch = self.locked_pitch + (ALTITUDE_GAIN * altitude_error)
        # Clamp pitch between -10 and +10 degrees
        return max(math.radians(-10.0), min(math.radians(10.0), pitch))

    def _pub_attitude(self, roll_rad, pitch_rad, yaw_rad, thrust):
        q = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        msg = VehicleAttitudeSetpoint()
        msg.timestamp   = int(self.get_clock().now().nanoseconds / 1000)
        msg.q_d         = [float(v) for v in q]
        msg.thrust_body = [thrust, 0.0, 0.0]
        self.attitude_pub.publish(msg)

    def _switch_offboard(self):
        self._send_cmd(CMD_SET_MODE,
                       p1=float(MAV_MODE_FLAG_CUSTOM),
                       p2=float(MAIN_MODE_OFFBOARD))

    def _switch_loiter(self):
        self._send_cmd(CMD_SET_MODE,
                       p1=float(MAV_MODE_FLAG_CUSTOM),
                       p2=float(MAIN_MODE_AUTO),
                       p3=float(SUB_MODE_AUTO_LOITER))


    # ── DAA loop (10 Hz) ──────────────────────────────────
    def _daa_loop(self):
        if self._vehicle_type != VEHICLE_TYPE_FW:
            return
        if not all([self._obs_pos, self._obs_vel,
                    self._intr_pos, self._intr_vel]):
            return

        evasive_hdg, current_hdg, dist, state, strat = compute_evasive_heading(
            self._obs_pos, self._obs_vel,
            self._intr_pos, self._intr_vel,
            self._strategy,
        )
        self._last_daa = (evasive_hdg, current_hdg, dist, state, strat)
        print(f"Evasive: {evasive_hdg}, Current: {current_hdg}, "
              f"Dist: {dist:.1f}, State: {state}")

        # ── Compute control outputs (always, so they appear in the log) ──
        airspeed   = self.current_airspeed if self.current_airspeed > 1.0 \
                     else TARGET_AIRSPEED
        thrust_cmd = self._compute_thrust()
        pitch_cmd  = self._compute_pitch()

        bank_cmd   = float("nan")
        evasive_log = float("nan")

        if evasive_hdg is not None:
            evasive_log = evasive_hdg
            bank_cmd, _, _ = calc_bank_for_heading(
                current_heading_deg=current_hdg,
                desired_heading_deg=evasive_hdg,
                airspeed_ms=airspeed,
            )


            # Check if we've reached the evasive heading
            heading_error = evasive_hdg - current_hdg
            while heading_error >  180.0: heading_error -= 360.0
            while heading_error < -180.0: heading_error += 360.0

            if abs(heading_error) < 5.0:
                # Close enough to target heading — level the wings
                bank_cmd = 0.0

            if self.nav_state != 14:           # not yet in OFFBOARD
                self._pub_offboard()
                self._pub_attitude(
                    roll_rad  = 0.0,
                    pitch_rad = pitch_cmd,
                    yaw_rad   = self.current_yaw,
                    thrust    = thrust_cmd,
                )
                self._switch_offboard()
            else:                              # already in OFFBOARD
                self._pub_attitude(
                    roll_rad  = math.radians(bank_cmd),
                    pitch_rad = pitch_cmd,
                    yaw_rad   = self.current_yaw,
                    thrust    = thrust_cmd,
                )
        else:
            self._switch_loiter()

        # ── Log row ───────────────────────────────────────────────────────
        row = {
            "timestamp_s":      time.time(),

            "obs_x_m":          self._obs_pos[0],
            "obs_y_m":          self._obs_pos[1],
            "obs_vx_ms":        self._obs_vel[0],
            "obs_vy_ms":        self._obs_vel[1],
            "obs_alt_m":        self.current_altitude,
            "obs_airspeed_ms":  self.current_airspeed,

            "intr_x_m":         self._intr_pos[0],
            "intr_y_m":         self._intr_pos[1],
            "intr_vx_ms":       self._intr_vel[0],
            "intr_vy_ms":       self._intr_vel[1],

            "separation_m":     dist,

            "daa_state":        state,
            "current_hdg_deg":  current_hdg,
            "evasive_hdg_deg":  evasive_log,
            "bank_cmd_deg":     bank_cmd,
            "heading_error_deg": heading_error if evasive_hdg is not None else float("nan"),

            "thrust_cmd":       thrust_cmd,
            "pitch_cmd_deg":    math.degrees(pitch_cmd),

            "nav_state":        self.nav_state,
            "in_offboard":      1 if self.nav_state == 14 else 0,
        }
        self._daa_logger.write(row)
        self._daa_logger.flush()   # remove this line if you prefer buffered I/O

    # ── Curses display (2 Hz, in-place) ──────────────────────────
    def _debug_print(self):
        scr = self._scr
        if scr is None:
            return

        evasive_hdg, current_hdg, dist, state, _ = (
            self._last_daa if self._last_daa
            else (None, 0.0, 0.0, "NO DATA", "")
        )

        vtol_map = {VEHICLE_TYPE_FW: "FW", VEHICLE_TYPE_MC: "MC",
                    VEHICLE_TYPE_VTOL: "TRANS"}
        vtol_str = vtol_map.get(self._vehicle_type, "?")

        eva_str  = f"{evasive_hdg:+7.1f}" if evasive_hdg is not None else "    N/A"

        C_NORMAL = curses.color_pair(1)
        C_ALERT  = curses.color_pair(2)
        C_FAULT  = curses.color_pair(3)

        is_danger = self._last_daa and "DANGER" in state
        daa_col   = C_FAULT if is_danger else (
                    C_ALERT if "ALERT" in (state or "") else C_NORMAL)

        try:
            scr.erase()
            W = curses.COLS - 1

            # ── Row 0: title ─────────────────────────────────────
            scr.addstr(0, 0, " RTA DAA MONITOR ", curses.A_REVERSE | curses.A_BOLD)
            scr.addstr(0, 18, " q=quit", C_NORMAL)

            # ── Row 2: Vehicle ───────────────────────────────────
            scr.addstr(2, 0, "VEHICLE", curses.A_BOLD)
            scr.addstr(3, 2, f"Type : {vtol_str:<6}", C_NORMAL)

            # ── Row 5: DAA ───────────────────────────────────────
            scr.addstr(5, 0, "DAA", curses.A_BOLD)
            scr.addstr(6, 2,  f"State    : {state:<28}", daa_col)
            scr.addstr(6, 44, f"Distance : {dist:7.1f} m", daa_col)
            scr.addstr(7, 2,  f"Heading  : {current_hdg:+7.1f} deg", C_NORMAL)
            scr.addstr(7, 44, f"Evasive  : {eva_str} deg", daa_col)

            # ── Row 9: Event log ─────────────────────────────────
            scr.addstr(9, 0, "EVENT LOG", curses.A_BOLD)
            for i, line in enumerate(self._event_log):
                scr.addstr(10 + i, 2, line[:W - 2], C_NORMAL)

            scr.refresh()
        except curses.error:
            pass


# =============================================================================
# MAIN
# =============================================================================
def _curses_main(stdscr, node):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE,  -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED,    -1)
    node._scr = stdscr
    node._log("RTA DAA monitor started")
    rclpy.spin(node)


def main():
    rclpy.init()
    node = RTANode()
    try:
        rclpy.spin(node)
        #curses.wrapper(_curses_main, node)
    except KeyboardInterrupt:
        pass
    finally:
        node._daa_logger.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()




#!/usr/bin/env python3
"""
RollRTA — Servo fault detection with two-layer RTA.

Detection layers
----------------
Layer 1  Gate + LLR   (original baseline logic)
    Sudden failures: z-score gate opens → LLR confirms faulty distribution.
    Fast response, ~1-2 s latency after gate duration.

Layer 2  CUSUM on Mahalanobis distance  (new)
    Gradual degradation: accumulates evidence across windows.
    Catches slow drift that never spikes the z-score gate.

Corrective actions
------------------
Fault confirmed  →  VTOL transition to MC (ailerons not needed in MC mode)
CUSUM building   →  Warn + increase log verbosity (no mode change yet)
"""

import math
import time
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from collections import deque

# Uncomment when running on real system:
from helper_functions import *
from PX4Vehicle import PX4Vehicle

WAYPOINTS = [
    {"lat": 37.411796, "lon": -121.997816, "alt": 50.0},
]

TAKEOFF_ALT          = 45.0
SERVO_TO_LOCK        = 0
ANGLE_TO_LOCK_DEG    = 20.0
FAILURE_START_SECS   = 150.0
BASELINE             = False

SET_CASE = 3  # 1=no failure no RTA, 2=failure no RTA, 3=failure+RTA
SIMULATE_FAILURE = SET_CASE in (2, 3)
RTA_ON           = SET_CASE == 3

# =============================================================================
# ROLL MONITOR PARAMETERS
# =============================================================================
ROLL_WINDOW_SIZE            = 5
ROLL_GATE_MEAN_DEG          = 0.0
ROLL_GATE_STD_DEG           = 1.5
ROLL_GATE_Z_THRESHOLD       = 5.0
ROLL_GATE_CLOSE_Z_THRESHOLD = 1.5
ROLL_SETTLING_TIME_S        = 1.10
ROLL_SETTLED_THRESHOLD_DEG  = 2.0
ROLL_MIN_GATE_DURATION      = 1.0
ROLL_NOMINAL_MEAN_DEG       = 0.0529
ROLL_NOMINAL_STD_DEG        = 4.2037
ROLL_FAULTY_MEAN_DEG        = 8.0   # TODO: update from locked-servo flight log
ROLL_FAULTY_STD_DEG         = 2.5   # TODO: update from locked-servo flight log

# =============================================================================
# PITCH MONITOR PARAMETERS
# =============================================================================
PITCH_WINDOW_SIZE            = 5
PITCH_SETTLING_TIME_S        = 1.10
PITCH_SETTLED_THRESHOLD_DEG  = 2.0
PITCH_MIN_GATE_DURATION      = 1.0
PITCH_SECONDARY_GATE_DEG     = 4.0
PITCH_NOMINAL_MEAN_DEG       = 0.5   # TODO: update after nominal calibration
PITCH_NOMINAL_STD_DEG        = 1.5   # TODO: update after nominal calibration
PITCH_FAULTY_MEAN_DEG        = 8.0   # TODO: update after faulty calibration
PITCH_FAULTY_STD_DEG         = 2.5   # TODO: update after faulty calibration

WINDOW_SIZE = 5

# =============================================================================
# CUSUM PARAMETERS
# =============================================================================
CUSUM_ALPHA = 0.3    # EMA smoothing — fast track for CUSUM input
CUSUM_K     = 1.0    # Allowance — ignore shifts smaller than this (in Mahal units)
                     # Rule of thumb: half the shift magnitude you care about detecting.
                     # Your Mahal dist under healthy is ~1.18; a fault shifts it to ~3-5.
                     # So k=1.0 means "alert me if the mean shifts by >2 Mahal units".
CUSUM_H     = 6.0    # Decision threshold — higher = fewer false alarms, slower detection.
                     # At k=1.0, h=6.0 gives ~1 false alarm per 500 samples under healthy.
CUSUM_MU_D  = 1.18   # Expected Mahalanobis distance under a healthy 2D Gaussian (theoretical).
                     # Verify empirically: np.mean([mahal(s) for s in healthy_flight_log])


# =============================================================================
# Helpers
# =============================================================================

def _mahalanobis(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    """Mahalanobis distance from x to distribution (mu, sigma_inv)."""
    d = x - mu
    return float(np.sqrt(d @ sigma_inv @ d))


def _axis_llr(x, mu_nom, sig_nom, mu_flt, sig_flt) -> float:
    """
    Log-likelihood ratio for a single axis under two Gaussian hypotheses.
    LLR > 0 means the faulty distribution is more likely than nominal.
    """
    return (
        math.log(sig_nom / sig_flt)
        + (x - mu_nom) ** 2 / (2.0 * sig_nom ** 2)
        - (x - mu_flt) ** 2 / (2.0 * sig_flt ** 2)
    )


# =============================================================================
# RollRTA node
# =============================================================================

class RollRTA(PX4Vehicle):

    def __init__(self):
        super().__init__()
        self.get_logger().info("Roll RTA node initialized")

        self.locked_servo = False
        self._error_window = deque(maxlen=WINDOW_SIZE)

        # ── Layer 1: Gate + LLR state (unchanged from baseline) ───────────────
        self._gate_open    = False
        self._gate_start_t = 0.0
        self._cond_a       = False
        self._cond_b       = False
        self._is_fault     = False
        self._llr_val      = None

        # ── Layer 2: CUSUM state ───────────────────────────────────────────────
        self._cusum_Sp      = 0.0   # upward cumulative sum (detects mean increase)
        self._cusum_Sn      = 0.0   # downward cumulative sum (detects mean decrease)
        self._cusum_faulted = False  # latched once CUSUM threshold crossed
        self._cusum_fault_t = None

        # EMA state for CUSUM input (smoothed, separate from window mean)
        self._ema_roll_err  = None
        self._ema_pitch_err = None

        # Healthy reference distribution P = N(mu, Sigma)
        # Built from your calibrated ROLL/PITCH_NOMINAL parameters.
        # Once you have real flight logs, replace with:
        #   data = np.load('healthy_errors.npy')  # shape (N, 2), degrees
        #   self._P_mu  = data.mean(axis=0)
        #   self._P_sigma = np.cov(data, rowvar=False) + 0.3 * np.eye(2)
        rho    = 0.2  # roll-pitch error correlation; update from calibration
        cov_rp = rho * ROLL_NOMINAL_STD_DEG * PITCH_NOMINAL_STD_DEG
        P_sigma = np.array([
            [ROLL_NOMINAL_STD_DEG ** 2 + 0.3, cov_rp],
            [cov_rp, PITCH_NOMINAL_STD_DEG ** 2 + 0.3],
        ])
        self._P_mu        = np.array([ROLL_NOMINAL_MEAN_DEG, PITCH_NOMINAL_MEAN_DEG])
        self._P_sigma     = P_sigma
        self._P_sigma_inv = np.linalg.inv(P_sigma)

        # Misc
        self._fault_logged = False
        self.time_started  = time.time()

        # ── Timers ─────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._tick)        # 10 Hz flight state machine
        if RTA_ON:
            self.get_logger().info("RTA logic enabled")
            if BASELINE:
                self.create_timer(0.05, self._rta_loop_baseline)  # 20 Hz
            else:
                self.create_timer(0.05, self._rta_loop)           # 20 Hz
        else:
            self.get_logger().info("RTA logic disabled — running open-loop")

    # =========================================================================
    # Flight state machine (unchanged)
    # =========================================================================

    def _tick(self):
        if SIMULATE_FAILURE and time.time() - self.time_started > FAILURE_START_SECS \
                and not self.locked_servo:
            lock_servo(SERVO_TO_LOCK, ANGLE_TO_LOCK_DEG * math.pi / 180, self.model)
            self.get_logger().warn(
                f"Simulated failure: servo {SERVO_TO_LOCK} locked at {ANGLE_TO_LOCK_DEG} deg"
            )
            self.locked_servo = True

        if self.state == "IDLE":
            self._arm()
            self.state = "ARMING"

        elif self.state == "ARMING":
            if self.armed:
                self.get_logger().info("Armed ✓")
                self._vtol_mc_takeoff()
                self.state = "VTOL_MC_TAKEOFF"

        elif self.state == "VTOL_MC_TAKEOFF":
            if self.local_alt >= TAKEOFF_ALT:
                self._vtol_mc_loiter()
                self.state = "VTOL_MC_LOITER"
                self.hold_until = time.time() + 5.0

        elif self.state == "VTOL_MC_LOITER":
            if time.time() >= self.hold_until:
                self._vtol_transition_to_fw()
                self.state = "VTOL_TRANSITION_FW"
                self.hold_until = time.time() + 10.0

        elif self.state == "VTOL_TRANSITION_FW":
            if time.time() >= self.hold_until:
                self.state = "FLYING_NORMAL"

        elif self.state == "FLYING_NORMAL":
            wp = WAYPOINTS[self.current_wp_index]
            self._fly_to_waypoint(wp)
            if time.time() >= self.time_started + 300.0:
                self.state = "VTOL_TRANSITION_MC"
                self._vtol_transition_to_mc()

        elif self.state == "VTOL_TRANSITION_MC":
            self._vtol_mc_land()
            self.state = "LANDING"

    # =========================================================================
    # Layer 1 + Layer 2 RTA loop
    # =========================================================================

    def _rta_loop(self):
        """
        Two-layer fault detection at 20 Hz.

        Layer 1 — Gate + LLR   : sudden hard failures (your existing logic)
        Layer 2 — CUSUM         : gradual drift / slow degradation (new)
        """

        # ── Guard ─────────────────────────────────────────────────────────────
        if (self._desired_roll  is None or self.current_roll  is None or
                self._desired_pitch is None or self.current_pitch is None):
            return

        # ── 1. Raw errors ─────────────────────────────────────────────────────
        err_roll_rad  = self._desired_roll  - self.current_roll
        err_pitch_rad = self._desired_pitch - self.current_pitch
        err_roll_deg  = math.degrees(err_roll_rad)
        err_pitch_deg = math.degrees(err_pitch_rad)

        # ── 2. EMA smoothing (CUSUM input) ────────────────────────────────────
        if self._ema_roll_err is None:
            # First sample: initialize EMA directly
            self._ema_roll_err  = err_roll_deg
            self._ema_pitch_err = err_pitch_deg
        else:
            self._ema_roll_err  = CUSUM_ALPHA * err_roll_deg  + (1 - CUSUM_ALPHA) * self._ema_roll_err
            self._ema_pitch_err = CUSUM_ALPHA * err_pitch_deg + (1 - CUSUM_ALPHA) * self._ema_pitch_err

        # ── 3. Sliding window (Layer 1 input, unchanged from baseline) ────────
        self._error_window.append([err_roll_rad, err_pitch_rad])
        if len(self._error_window) < WINDOW_SIZE:
            return  # not enough data yet

        n          = len(self._error_window)
        mean_roll  = sum(s[0] for s in self._error_window) / n
        mean_pitch = sum(s[1] for s in self._error_window) / n
        mean_roll_deg  = math.degrees(mean_roll)
        mean_pitch_deg = math.degrees(mean_pitch)

        # ── 4. Layer 2: CUSUM ─────────────────────────────────────────────────
        #
        # We run CUSUM on the Mahalanobis distance of the EMA-smoothed error
        # from the healthy reference P.  The Mahalanobis score compresses the
        # 2D [roll, pitch] error into a single scalar that accounts for the
        # correlation structure of healthy errors.
        #
        # CUSUM update:
        #   S+_t = max(0,  S+_{t-1} + (d_t - mu_d) - k)   upward shift
        #   S-_t = max(0,  S-_{t-1} - (d_t - mu_d) - k)   downward shift (sanity check)
        #
        # Alert when S+ > h  (distance has been growing — fault likely)
        #
        ema_vec = np.array([self._ema_roll_err, self._ema_pitch_err])
        d       = _mahalanobis(ema_vec, self._P_mu, self._P_sigma_inv)

        if not self._cusum_faulted:
            excess         = d - CUSUM_MU_D
            self._cusum_Sp = max(0.0, self._cusum_Sp + excess - CUSUM_K)
            self._cusum_Sn = max(0.0, self._cusum_Sn - excess - CUSUM_K)

            if self._cusum_Sp > CUSUM_H or self._cusum_Sn > CUSUM_H:
                self._cusum_faulted = True
                self.get_logger().warn(
                    f"[CUSUM] Gradual fault detected — "
                    f"S+={self._cusum_Sp:.2f}  S-={self._cusum_Sn:.2f}  "
                    f"d={d:.2f}  "
                    f"ema_roll={self._ema_roll_err:.2f}°  "
                    f"ema_pitch={self._ema_pitch_err:.2f}°"
                )

        # ── 5. Layer 1: Gate + LLR ────────────────────────────────────────────
        z_roll  = abs(mean_roll_deg  - ROLL_NOMINAL_MEAN_DEG)  / ROLL_NOMINAL_STD_DEG
        z_pitch = abs(mean_pitch_deg - PITCH_NOMINAL_MEAN_DEG) / PITCH_NOMINAL_STD_DEG

        gate_open_trigger  = (z_roll  > ROLL_GATE_Z_THRESHOLD) or  \
                             (z_pitch > ROLL_GATE_Z_THRESHOLD)
        gate_close_trigger = (z_roll  <= ROLL_GATE_CLOSE_Z_THRESHOLD) and \
                             (z_pitch <= ROLL_GATE_CLOSE_Z_THRESHOLD)

        if not self._gate_open:
            if gate_open_trigger:
                self._gate_open    = True
                self._gate_start_t = time.monotonic()
                self._cond_a = self._cond_b = self._is_fault = False
                self.get_logger().warn(
                    f"[Gate] Opened — "
                    f"z_roll={z_roll:.2f}  z_pitch={z_pitch:.2f}  "
                    f"roll={mean_roll_deg:.2f}°  pitch={mean_pitch_deg:.2f}°"
                )
            elif gate_close_trigger:
                self._llr_val  = None
                self._is_fault = False

        else:
            elapsed = time.monotonic() - self._gate_start_t

            if elapsed >= max(ROLL_MIN_GATE_DURATION, PITCH_MIN_GATE_DURATION):
                llr_roll  = _axis_llr(mean_roll_deg,
                                      ROLL_NOMINAL_MEAN_DEG,  ROLL_NOMINAL_STD_DEG,
                                      ROLL_FAULTY_MEAN_DEG,   ROLL_FAULTY_STD_DEG)
                llr_pitch = _axis_llr(mean_pitch_deg,
                                      PITCH_NOMINAL_MEAN_DEG, PITCH_NOMINAL_STD_DEG,
                                      PITCH_FAULTY_MEAN_DEG,  PITCH_FAULTY_STD_DEG)
                self._llr_val = llr_roll + llr_pitch
                self._cond_a  = self._llr_val > 0
                self._cond_b  = elapsed >= max(ROLL_SETTLING_TIME_S, PITCH_SETTLING_TIME_S)

                settled = (abs(mean_roll_deg)  < ROLL_SETTLED_THRESHOLD_DEG and
                           abs(mean_pitch_deg) < PITCH_SETTLED_THRESHOLD_DEG)

                if settled:
                    self.get_logger().info(
                        f"[Gate] Closed — settled in {elapsed:.1f}s  "
                        f"LLR={self._llr_val:.2f}"
                    )
                    self._gate_open = False
                    self._llr_val   = None

                elif self._cond_a and self._cond_b and not self._is_fault:
                    self.get_logger().warn(
                        f"[Gate+LLR] Sudden fault detected — "
                        f"LLR={self._llr_val:.2f}  "
                        f"roll={mean_roll_deg:.2f}°  pitch={mean_pitch_deg:.2f}°"
                    )
                    self._is_fault = True

        # ── 6. Corrective action ───────────────────────────────────────────────
        self._corrective_action(d)

    # =========================================================================
    # Corrective actions
    # =========================================================================

    def _corrective_action(self, mahal_dist: float):
        """
        Priority-ordered response based on combined detector state.

          Priority 1 — fault confirmed by Gate+LLR *or* CUSUM
          Priority 2 — CUSUM accumulating (> 60% of threshold) — warn only
          Priority 3 — healthy — no action
        """
        fault_confirmed = self._is_fault or self._cusum_faulted
        cusum_building  = (not self._cusum_faulted and
                           max(self._cusum_Sp, self._cusum_Sn) > CUSUM_H * 0.6)

        if fault_confirmed:
            self._action_fault_confirmed()
        elif cusum_building:
            self._action_cusum_warning(mahal_dist)

    def _action_fault_confirmed(self):
        """
        Hard fault — transition to VTOL MC (multicopter mode doesn't use ailerons).
        Only acts when in FLYING_NORMAL; other states handle themselves.
        """
        if self.state != "FLYING_NORMAL":
            return

        if not self._fault_logged:
            detector = "Gate+LLR" if self._is_fault else "CUSUM"
            elapsed  = time.time() - self.time_started
            self.get_logger().error(
                f"[RTA] FAULT CONFIRMED ({detector}) at t={elapsed:.1f}s — "
                f"transitioning to VTOL MC  "
                f"LLR={self._llr_val}  "
                f"S+={self._cusum_Sp:.2f}  S-={self._cusum_Sn:.2f}"
            )
            self._fault_logged = True

        self._vtol_transition_to_mc()
        self.state = "VTOL_TRANSITION_MC"

    def _action_cusum_warning(self, mahal_dist: float):
        """
        CUSUM is building but hasn't latched yet.
        Log at warn level. In a production system you might also
        tighten gain margins or increase autopilot damping here.
        """
        self.get_logger().warn(
            f"[RTA] CUSUM building — "
            f"S+={self._cusum_Sp:.2f}  S-={self._cusum_Sn:.2f}  "
            f"(h={CUSUM_H})  d={mahal_dist:.2f}  "
            f"ema_roll={self._ema_roll_err:.2f}°  "
            f"ema_pitch={self._ema_pitch_err:.2f}°"
        )

    # =========================================================================
    # Baseline loop (original — kept for comparison runs)
    # =========================================================================

    def _rta_loop_baseline(self):
        error_roll  = self._desired_roll  - self.current_roll  \
                      if self._desired_roll  is not None and self.current_roll  is not None else None
        error_pitch = self._desired_pitch - self.current_pitch \
                      if self._desired_pitch is not None and self.current_pitch is not None else None

        if error_roll is None or error_pitch is None:
            return

        self._error_window.append([error_roll, error_pitch])
        if len(self._error_window) < WINDOW_SIZE:
            return

        n          = len(self._error_window)
        mean_roll  = sum(s[0] for s in self._error_window) / n
        mean_pitch = sum(s[1] for s in self._error_window) / n
        mean_roll_deg  = math.degrees(mean_roll)
        mean_pitch_deg = math.degrees(mean_pitch)

        z_roll  = abs(mean_roll_deg  - ROLL_NOMINAL_MEAN_DEG)  / ROLL_NOMINAL_STD_DEG
        z_pitch = abs(mean_pitch_deg - PITCH_NOMINAL_MEAN_DEG) / PITCH_NOMINAL_STD_DEG

        gate_open_trigger  = (z_roll > ROLL_GATE_Z_THRESHOLD) or (z_pitch > ROLL_GATE_Z_THRESHOLD)
        gate_close_trigger = (z_roll <= ROLL_GATE_CLOSE_Z_THRESHOLD) and (z_pitch <= ROLL_GATE_CLOSE_Z_THRESHOLD)

        if not self._gate_open:
            if gate_open_trigger:
                self._gate_open    = True
                self._gate_start_t = time.monotonic()
                self._cond_a = self._cond_b = self._is_fault = False
                self.get_logger().warn(
                    f"Multivariate gate opened — "
                    f"z_roll={z_roll:.2f} z_pitch={z_pitch:.2f} "
                    f"roll_err={mean_roll_deg:.2f}° pitch_err={mean_pitch_deg:.2f}°"
                )
            elif gate_close_trigger:
                self._llr_val  = None
                self._is_fault = False
            return

        elapsed = time.monotonic() - self._gate_start_t
        if elapsed < max(ROLL_MIN_GATE_DURATION, PITCH_MIN_GATE_DURATION):
            return

        llr_roll  = _axis_llr(mean_roll_deg,
                               ROLL_NOMINAL_MEAN_DEG,  ROLL_NOMINAL_STD_DEG,
                               ROLL_FAULTY_MEAN_DEG,   ROLL_FAULTY_STD_DEG)
        llr_pitch = _axis_llr(mean_pitch_deg,
                               PITCH_NOMINAL_MEAN_DEG, PITCH_NOMINAL_STD_DEG,
                               PITCH_FAULTY_MEAN_DEG,  PITCH_FAULTY_STD_DEG)
        self._llr_val = llr_roll + llr_pitch
        self._cond_a  = self._llr_val > 0

        settled = (abs(mean_roll_deg) < ROLL_SETTLED_THRESHOLD_DEG and
                   abs(mean_pitch_deg) < PITCH_SETTLED_THRESHOLD_DEG)
        if settled:
            self.get_logger().info(
                f"Multivariate gate closed — settled in {elapsed:.1f}s "
                f"(LLR={self._llr_val:.2f})"
            )
            self._gate_open = False
            self._llr_val   = None
            return

        self._cond_b = elapsed >= max(ROLL_SETTLING_TIME_S, PITCH_SETTLING_TIME_S)

        if self._cond_a and self._cond_b and not self._is_fault:
            self.get_logger().warn(
                f"MULTIVARIATE FAULT DETECTED — LLR={self._llr_val:.2f} "
                f"roll_err={mean_roll_deg:.2f}° pitch_err={mean_pitch_deg:.2f}°"
            )
            self._is_fault = True

    def _write_log(self, roll_err, roll_elapsed, roll_llr_v):
        pitch_ela = (time.monotonic() - self._pitch_gate_start_t
                     if self._pitch_gate_start_t else float("nan"))
        dp = self._desired_pitch
        ap = self._actual_pitch
        self._csv_logger.write({
            "timestamp_s":             time.time(),
            "actual_roll_deg":         math.degrees(self._actual_roll),
            "desired_roll_deg":        math.degrees(self._desired_roll),
            "raw_roll_error_deg":      math.degrees(roll_err),
            "smoothed_roll_error_deg": math.degrees(self._roll_smoothed_err),
            "roll_gate_open":          int(self._roll_gate_open),
            "roll_gate_elapsed_s":     roll_elapsed,
            "roll_llr":                roll_llr_v if roll_llr_v is not None
                                       else float("nan"),
            "roll_cond_a":             int(self._roll_cond_a),
            "roll_cond_b":             int(self._roll_cond_b),
            "roll_is_fault":           int(self._roll_is_fault),
            "actual_pitch_deg":        math.degrees(ap) if ap is not None else float("nan"),
            "desired_pitch_deg":       math.degrees(dp) if dp is not None else float("nan"),
            "raw_pitch_error_deg":     math.degrees(dp-ap) if dp is not None
                                       and ap is not None else float("nan"),
            "smoothed_pitch_error_deg": math.degrees(self._pitch_smoothed_err),
            "pitch_gate_open":         int(self._pitch_gate_open),
            "pitch_gate_elapsed_s":    pitch_ela,
            "pitch_llr":               math.degrees(self._pitch_llr_val) if self._pitch_llr_val is not None
                                       else float("nan"),
            "pitch_cond_a":            int(self._pitch_cond_a),
            "pitch_cond_b":            int(self._pitch_cond_b),
            "pitch_is_fault":          int(self._pitch_is_fault),
            "fault_injected":          int(self._sw_fault_active),
            "physical_fault_active":   int(self._physical_fault),
            "nav_state":               self._nav_state,
        })
# =============================================================================
# MAIN
# =============================================================================

def main():
    rclpy.init()
    node = DAARTA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()