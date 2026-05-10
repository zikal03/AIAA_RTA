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
ANGLE_TO_LOCK_DEG    = 35.0
FAILURE_START_SECS   = 150.0
BASELINE             = True

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
            #self._action_fault_confirmed()
        

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
    node = RollRTA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()