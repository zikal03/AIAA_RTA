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

Fixes applied
-------------
1. Gate close trigger moved into the gate-open branch so it can actually
   close the gate while it is open (both loops).
2. _is_fault, _cond_a, _cond_b all reset when gate closes (both loops).
3. _rta_loop used ROLL_GATE_Z_THRESHOLD for pitch — changed to PITCH_GATE_Z_THRESHOLD.
4. _rta_loop gate-close path now sets _gate_open=False and resets all conditions.
5. Settled check re-enabled in _rta_loop_baseline and both loops now behave
   identically: gate closes via gate_close_trigger OR settled, whichever
   comes first.
6. "Gate closed" log message in baseline only fires when gate was actually open.
7. VTOL_TRANSITION_MC now waits for transition to complete before landing.
"""

import csv
import math
import os
import time
import queue
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
ANGLE_TO_LOCK_DEG    = 15
FAILURE_START_SECS   = 150
FLIGHT_END           = 300
VTOL_TRANSITION_WAIT = 10.0   # seconds to wait for MC transition before landing
BASELINE             = True

SET_CASE = 3  # 1=no failure no RTA, 2=failure no RTA, 3=failure+RTA
SIMULATE_FAILURE = SET_CASE in (2, 3)
RTA_ON           = SET_CASE == 3

LOG_NAME = "Test_1"  # change before each run

# =============================================================================
# CSV LOGGER
# =============================================================================

class Logger:
    """
    Writes one row per RTA loop tick to a timestamped CSV file.
    All angles in degrees.
    """
    FIELDS = [
        "timestamp_s",
        "actual_roll_deg",
        "desired_roll_deg",
        "roll_error_deg",
        "actual_pitch_deg",
        "desired_pitch_deg",
        "pitch_error_deg",
        "llr_val",
        "cond_a",
        "cond_b",
        "rta_fault_confirmed",
        "servo_failed",
        "angle_locked"
    ]

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{LOG_NAME}_{ts}.csv")
        self._f    = open(path, "w", newline="")
        self._w    = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._path = path
        print(f"[Logger] Logging to {path}")

    def write(self, row: dict):
        """Write one row; missing keys default to empty string."""
        self._w.writerow({f: row.get(f, "") for f in self.FIELDS})

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()
        print(f"[Logger] Closed {self._path}")

# =============================================================================
# ROLL MONITOR PARAMETERS
# =============================================================================
ROLL_GATE_Z_THRESHOLD       = 5.0
ROLL_GATE_CLOSE_Z_THRESHOLD = 1.5
ROLL_SETTLING_TIME_S        = 1.3
ROLL_MIN_GATE_DURATION      = 1.0
ROLL_NOMINAL_MEAN_DEG       = -0.2382
ROLL_NOMINAL_STD_DEG        =  1.4445
ROLL_FAULTY_MEAN_DEG        =  5.2679
ROLL_FAULTY_STD_DEG         =  3.1269
ROLL_SETTLED_THRESHOLD_DEG  = ROLL_NOMINAL_MEAN_DEG + ROLL_NOMINAL_STD_DEG

# =============================================================================
# PITCH MONITOR PARAMETERS
# =============================================================================
PITCH_GATE_Z_THRESHOLD       = 5.0   # FIX 3: was accidentally using ROLL threshold in _rta_loop
PITCH_GATE_CLOSE_Z_THRESHOLD = 1.5
PITCH_SETTLING_TIME_S        = 1.3
PITCH_MIN_GATE_DURATION      = 1.0
PITCH_NOMINAL_MEAN_DEG       =  0.3037
PITCH_NOMINAL_STD_DEG        =  1.1359
PITCH_FAULTY_MEAN_DEG        =  0.1874
PITCH_FAULTY_STD_DEG         =  0.8273
PITCH_SETTLED_THRESHOLD_DEG  = PITCH_NOMINAL_MEAN_DEG + PITCH_NOMINAL_STD_DEG

WINDOW_SIZE = 5

# =============================================================================
# CUSUM PARAMETERS
# =============================================================================
CUSUM_ALPHA = 0.3
CUSUM_K     = 1.0
CUSUM_H     = 6.0
CUSUM_MU_D  = 1.18


# =============================================================================
# Helpers
# =============================================================================

def _mahalanobis(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    d = x - mu
    return float(np.sqrt(d @ sigma_inv @ d))


def _axis_llr(x, mu_nom, sig_nom, mu_flt, sig_flt) -> float:
    return (
        math.log(sig_nom / sig_flt)
        + (x - mu_nom) ** 2 / (2.0 * sig_nom ** 2)
        - (x - mu_flt) ** 2 / (2.0 * sig_flt ** 2)
    )


def _reset_gate(node):
    """
    Helper: clear all gate + LLR state.
    Called whenever the gate closes, regardless of reason.
    """
    node._gate_open = False
    node._llr_val   = None
    node._cond_a    = False   # FIX 2: reset conditions on gate close
    node._cond_b    = False   # FIX 2
    node._is_fault  = False   # FIX 2


# =============================================================================
# RollRTA node
# =============================================================================

class RollRTA(PX4Vehicle):

    def __init__(self):
        super().__init__()
        self.get_logger().info("Roll RTA node initialized")

        self.locked_servo = False  # True = real servo failure active

        self._error_window = deque(maxlen=WINDOW_SIZE)

        # ── Layer 1: Gate + LLR state ─────────────────────────────────────────
        self._gate_open    = False
        self._gate_start_t = 0.0
        self._cond_a       = False
        self._cond_b       = False
        self._is_fault     = False  # RTA output: fault confirmed
        self._llr_val      = None

        # ── Layer 2: CUSUM state ───────────────────────────────────────────────
        self._cusum_Sp      = 0.0
        self._cusum_Sn      = 0.0
        self._cusum_faulted = False
        self._cusum_fault_t = None

        self._ema_roll_err  = None
        self._ema_pitch_err = None

        rho    = 0.2
        cov_rp = rho * ROLL_NOMINAL_STD_DEG * PITCH_NOMINAL_STD_DEG
        P_sigma = np.array([
            [ROLL_NOMINAL_STD_DEG ** 2 + 0.3, cov_rp],
            [cov_rp, PITCH_NOMINAL_STD_DEG ** 2 + 0.3],
        ])
        self._P_mu        = np.array([ROLL_NOMINAL_MEAN_DEG, PITCH_NOMINAL_MEAN_DEG])
        self._P_sigma     = P_sigma
        self._P_sigma_inv = np.linalg.inv(P_sigma)

        self._fault_logged       = False
        self._transition_mc_t    = None   # FIX 7: timestamp when MC transition started
        self.time_started        = time.time()

        # ── CSV logger ────────────────────────────────────────────────────────
        self._csv_logger = Logger()

        # ── CSV log writer thread ─────────────────────────────────────────────
        self._log_queue  = queue.Queue()
        self._log_thread = threading.Thread(target=self._log_writer, daemon=True)
        self._log_thread.start()

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._tick)
        if RTA_ON:
            self.get_logger().info("RTA logic enabled")
            if BASELINE:
                self.create_timer(0.05, self._rta_loop_baseline)  # 20 Hz
            else:
                self.create_timer(0.05, self._rta_loop)           # 20 Hz
        else:
            self.get_logger().info("RTA logic disabled — running open-loop")

    # =========================================================================
    # State machine
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
            if time.time() >= self.time_started + FLIGHT_END:
                self._vtol_transition_to_mc()
                self._transition_mc_t = time.time()   # FIX 7: record when transition started
                self.state = "VTOL_TRANSITION_MC"

        elif self.state == "VTOL_TRANSITION_MC":
            # FIX 7: wait for the transition to complete before commanding land
            if self._transition_mc_t is not None and \
                    time.time() - self._transition_mc_t >= VTOL_TRANSITION_WAIT:
                self.get_logger().info("MC transition complete — commanding land")
                self._vtol_mc_land()
                self.state = "LANDING"

    # =========================================================================
    # Layer 1 + Layer 2 RTA loop
    # =========================================================================

    def _rta_loop(self):
        if (self._desired_roll  is None or self.current_roll  is None or
                self._desired_pitch is None or self.current_pitch is None):
            return

        err_roll_rad  = self._desired_roll  - self.current_roll
        err_pitch_rad = self._desired_pitch - self.current_pitch
        err_roll_deg  = math.degrees(err_roll_rad)
        err_pitch_deg = math.degrees(err_pitch_rad)

        # EMA smoothing (CUSUM input)
        if self._ema_roll_err is None:
            self._ema_roll_err  = err_roll_deg
            self._ema_pitch_err = err_pitch_deg
        else:
            self._ema_roll_err  = CUSUM_ALPHA * err_roll_deg  + (1 - CUSUM_ALPHA) * self._ema_roll_err
            self._ema_pitch_err = CUSUM_ALPHA * err_pitch_deg + (1 - CUSUM_ALPHA) * self._ema_pitch_err

        # Sliding window (Layer 1 input)
        self._error_window.append([err_roll_rad, err_pitch_rad])
        if len(self._error_window) < WINDOW_SIZE:
            return

        n = len(self._error_window)
        mean_roll_deg  = math.degrees(sum(s[0] for s in self._error_window) / n)
        mean_pitch_deg = math.degrees(sum(s[1] for s in self._error_window) / n)

        # ── Layer 2: CUSUM ────────────────────────────────────────────────────
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

        # ── Layer 1: Gate + LLR ───────────────────────────────────────────────
        z_roll  = abs(mean_roll_deg  - ROLL_NOMINAL_MEAN_DEG)  / ROLL_NOMINAL_STD_DEG
        z_pitch = abs(mean_pitch_deg - PITCH_NOMINAL_MEAN_DEG) / PITCH_NOMINAL_STD_DEG

        gate_open_trigger  = (z_roll  > ROLL_GATE_Z_THRESHOLD) or \
                             (z_pitch > PITCH_GATE_Z_THRESHOLD)   # FIX 3: correct variable
        gate_close_trigger = (z_roll  <= ROLL_GATE_CLOSE_Z_THRESHOLD) and \
                             (z_pitch <= PITCH_GATE_CLOSE_Z_THRESHOLD)

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
        else:
            # ── Gate is open ──────────────────────────────────────────────────
            elapsed = time.monotonic() - self._gate_start_t

            # FIX 1+4: check close trigger while gate is open, reset everything
            if gate_close_trigger:
                self.get_logger().info(
                    f"[Gate] Closed via z-score — "
                    f"z_roll={z_roll:.2f}  z_pitch={z_pitch:.2f}  "
                    f"elapsed={elapsed:.1f}s"
                )
                _reset_gate(self)
                self._write_log()
                self._corrective_action(d)
                return

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

                # FIX 5: settled check active and consistent with baseline loop
                if settled:
                    self.get_logger().info(
                        f"[Gate] Closed — settled in {elapsed:.1f}s  "
                        f"LLR={self._llr_val:.2f}"
                    )
                    _reset_gate(self)  # FIX 2: reset all conditions on settle
                elif self._cond_a and self._cond_b and not self._is_fault:
                    self.get_logger().warn(
                        f"[Gate+LLR] Sudden fault detected — "
                        f"LLR={self._llr_val:.2f}  "
                        f"roll={mean_roll_deg:.2f}°  pitch={mean_pitch_deg:.2f}°"
                    )
                    self._is_fault = True

        self._corrective_action(d)
        self._write_log()

    # =========================================================================
    # Baseline loop
    # =========================================================================

    def _rta_loop_baseline(self):
        if (self._desired_roll  is None or self.current_roll  is None or
                self._desired_pitch is None or self.current_pitch is None):
            return

        error_roll  = self._desired_roll  - self.current_roll
        error_pitch = self._desired_pitch - self.current_pitch

        self._error_window.append([error_roll, error_pitch])

        if len(self._error_window) < WINDOW_SIZE:
            return

        n = len(self._error_window)
        mean_roll_deg  = math.degrees(sum(s[0] for s in self._error_window) / n)
        mean_pitch_deg = math.degrees(sum(s[1] for s in self._error_window) / n)

        z_roll  = abs(mean_roll_deg  - ROLL_NOMINAL_MEAN_DEG)  / ROLL_NOMINAL_STD_DEG
        z_pitch = abs(mean_pitch_deg - PITCH_NOMINAL_MEAN_DEG) / PITCH_NOMINAL_STD_DEG

        gate_open_trigger  = (z_roll > ROLL_GATE_Z_THRESHOLD) or (z_pitch > PITCH_GATE_Z_THRESHOLD)
        gate_close_trigger = (z_roll <= ROLL_GATE_CLOSE_Z_THRESHOLD) and (z_pitch <= PITCH_GATE_CLOSE_Z_THRESHOLD)

        if not self._gate_open:
            if gate_open_trigger:
                self._gate_open    = True
                self._gate_start_t = time.monotonic()
                self._cond_a = self._cond_b = self._is_fault = False
                self.get_logger().warn(
                    f"[Gate] Opened — "
                    f"z_roll={z_roll:.2f} z_pitch={z_pitch:.2f} "
                    f"roll_err={mean_roll_deg:.2f}° pitch_err={mean_pitch_deg:.2f}°"
                )
            self._write_log()
            return

        # ── Gate is open ──────────────────────────────────────────────────────
        elapsed = time.monotonic() - self._gate_start_t

        # FIX 1: gate close trigger now checked while gate is open
        if gate_close_trigger:
            # FIX 6: log only fires here, when gate was actually open
            self.get_logger().info(
                f"[Gate] Closed via z-score — "
                f"z_roll={z_roll:.2f} z_pitch={z_pitch:.2f} "
                f"elapsed={elapsed:.1f}s"
            )
            _reset_gate(self)  # FIX 2: resets _gate_open, _is_fault, _cond_a, _cond_b, _llr_val
            self._write_log()
            return

        if elapsed < max(ROLL_MIN_GATE_DURATION, PITCH_MIN_GATE_DURATION):
            self._write_log()
            return

        # Gate open + min duration elapsed — run LLR
        llr_roll  = _axis_llr(mean_roll_deg,
                               ROLL_NOMINAL_MEAN_DEG,  ROLL_NOMINAL_STD_DEG,
                               ROLL_FAULTY_MEAN_DEG,   ROLL_FAULTY_STD_DEG)
        llr_pitch = _axis_llr(mean_pitch_deg,
                               PITCH_NOMINAL_MEAN_DEG, PITCH_NOMINAL_STD_DEG,
                               PITCH_FAULTY_MEAN_DEG,  PITCH_FAULTY_STD_DEG)
        self._llr_val = llr_roll + llr_pitch
        self._cond_a  = self._llr_val > 0
        self._cond_b  = elapsed >= max(ROLL_SETTLING_TIME_S, PITCH_SETTLING_TIME_S)

        settled = (abs(mean_roll_deg) < ROLL_SETTLED_THRESHOLD_DEG and
                   abs(mean_pitch_deg) < PITCH_SETTLED_THRESHOLD_DEG)

        # FIX 5: settled check re-enabled and consistent with _rta_loop
        if settled:
            self.get_logger().info(
                f"[Gate] Closed — settled in {elapsed:.1f}s "
                f"LLR={self._llr_val:.2f}"
            )
            _reset_gate(self)  # FIX 2
            self._write_log()
            return

        if self._cond_a and self._cond_b and not self._is_fault:
            self.get_logger().warn(
                f"[Gate+LLR] Fault detected — LLR={self._llr_val:.2f} "
                f"roll_err={mean_roll_deg:.2f}° pitch_err={mean_pitch_deg:.2f}°"
            )
            self._is_fault = True

        self._write_log()

    # =========================================================================
    # Corrective actions
    # =========================================================================

    def _corrective_action(self, mahal_dist: float):
        fault_confirmed = self._is_fault or self._cusum_faulted
        cusum_building  = (not self._cusum_faulted and
                           max(self._cusum_Sp, self._cusum_Sn) > CUSUM_H * 0.6)

        if fault_confirmed:
            self._action_fault_confirmed()
        elif cusum_building:
            self._action_cusum_warning(mahal_dist)

    def _action_fault_confirmed(self):
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
        self._transition_mc_t = time.time()   # FIX 7: record transition start
        self.state = "VTOL_TRANSITION_MC"

    def _action_cusum_warning(self, mahal_dist: float):
        self.get_logger().warn(
            f"[RTA] CUSUM building — "
            f"S+={self._cusum_Sp:.2f}  S-={self._cusum_Sn:.2f}  "
            f"(h={CUSUM_H})  d={mahal_dist:.2f}  "
            f"ema_roll={self._ema_roll_err:.2f}°  "
            f"ema_pitch={self._ema_pitch_err:.2f}°"
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_writer(self):
        """Background thread: drains the queue and writes rows to CSV."""
        while True:
            row = self._log_queue.get()
            if row is None:
                break
            self._csv_logger.write(row)

    def _write_log(self):
        """Non-blocking: enqueues a row for the background writer thread."""
        self._log_queue.put_nowait({
            "timestamp_s":         time.time(),
            "actual_roll_deg":     math.degrees(self.current_roll),
            "desired_roll_deg":    math.degrees(self._desired_roll),
            "roll_error_deg":      math.degrees(self._desired_roll - self.current_roll)
                                   if self._desired_roll is not None and self.current_roll is not None
                                   else float("nan"),
            "actual_pitch_deg":    math.degrees(self.current_pitch),
            "desired_pitch_deg":   math.degrees(self._desired_pitch),
            "pitch_error_deg":     math.degrees(self._desired_pitch - self.current_pitch)
                                   if self._desired_pitch is not None and self.current_pitch is not None
                                   else float("nan"),
            "llr_val":             self._llr_val if self._llr_val is not None else float("nan"),
            "cond_a":              self._cond_a,
            "cond_b":              self._cond_b,
            "rta_fault_confirmed": self._is_fault,
            "servo_failed":        int(self.locked_servo),
            "angle_locked":        ANGLE_TO_LOCK_DEG,
        })

# =============================================================================
# MAIN
# =============================================================================

def main():
    rclpy.init()
    node = RollRTA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._log_queue.put(None)
        node._log_thread.join()
        node._csv_logger.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
