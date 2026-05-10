#!/usr/bin/env python3

import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from helper_functions import *

from px4_msgs.msg import (
    VehicleCommand,
    VehicleStatus,
    VtolVehicleStatus,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleAttitude,
)

from pymavlink import mavutil
from collections import deque

class PX4Vehicle(Node):
    def __init__(self, namespace="px4_1", model="standard_vtol_1"):
        super().__init__(f"{namespace}_vehicle")

        # PX4 constants
        self.CMD_ARM           = 400    # MAV_CMD_COMPONENT_ARM_DISARM
        self.CMD_TAKEOFF       = 84     # MAV_CMD_NAV_VTOL_TAKEOFF
        self.CMD_DO_REPOSITION = 192     # MAV_CMD_NAV_WAYPOINT — not suitable for real-time control
        self.CMD_VTOL_TRANS    = 3000   # MAV_CMD_DO_VTOL_TRANSITION
        self.CMD_SET_MODE      = 176    # MAV_CMD_DO_SET_MODE

        # Mode flags
        self.MAV_MODE_FLAG_CUSTOM = 1

        # Main modes
        self.MAIN_MODE_AUTO = 4

        # Sub modes
        self.SUB_MODE_AUTO_TAKEOFF = 2
        self.SUB_MODE_AUTO_LOITER  = 3
        self.SUB_MODE_AUTO_RTL     = 5
        self.SUB_MODE_AUTO_LAND    = 6

        self.ARMING_STATE_ARMED = 2

        self.model = model  # Gazebo model name, used for servo locking in helper function

        # -- telemetry state
        self.armed      = False
        self.vtol_state = 0
        self.local_alt  = 0.0
        self.lat        = 0.0
        self.lon        = 0.0
        self.alt_amsl   = 0.0   # current AMSL altitude
        self.home_amsl  = None  # AMSL of ground at home — captured before takeoff

        # -- mission state
        self.state      = "IDLE"
        self.current_wp_index = 0
        self.lap_count  = 0
        self.hold_until = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -- publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, f"/{namespace}/fmu/in/vehicle_command", qos, )

        # -- subscribers
        self.create_subscription(VehicleStatus, f"/{namespace}/fmu/out/vehicle_status_v4", self._status_cb, qos, )
        self.create_subscription(VtolVehicleStatus, f"/{namespace}/fmu/out/vtol_vehicle_status",  self._vtol_cb, qos, )
        self.create_subscription(VehicleGlobalPosition, f"/{namespace}/fmu/out/vehicle_global_position",self._global_pos_cb, qos,)
        self.create_subscription(VehicleLocalPosition, f"/{namespace}/fmu/out/vehicle_local_position_v1", self._local_pos_cb, qos,)
        self.create_subscription(VehicleAttitude, f"/{namespace}/fmu/out/vehicle_attitude", self._attitude_cb, qos)

        self.get_logger().info(f"{namespace} Node started -- waiting 1s for topics...")
        time.sleep(1.0)

        # -- pymavlink connection for ATTITUDE_TARGET
        # Thread-safe storage for desired attitude from ATTITUDE_TARGET
        self._att_target_lock = threading.Lock()
        self._desired_roll  = None   # radians
        self._desired_pitch = None   # radians
        self._desired_yaw   = None   # radians

        port = 14540 + int(namespace.split("_")[-1])  # Extract instance number from namespace
        self._mav = mavutil.mavlink_connection(f'udp:0.0.0.0:{port}', source_system=255, source_component=0,)
        
        self.get_logger().info("Waiting for MAVLink heartbeat (up to 10 s)…")
        hb = self._mav.wait_heartbeat(timeout=10)
        if hb is not None:
            self.get_logger().info(
                f"MAVLink connected — system {self._mav.target_system}, "
                f"component {self._mav.target_component}"
            )
        else:
            self.get_logger().warn(
                "MAVLink heartbeat timed out — ATTITUDE_TARGET will be unavailable"
            )

        self._mav_thread = threading.Thread(target=self._mav_reader, daemon=True)
        self._mav_thread.start()

    # =========================================================================
    # Subscribers
    # =========================================================================

    def _status_cb(self, msg: VehicleStatus):
        self.armed = (msg.arming_state == self.ARMING_STATE_ARMED)

    def _vtol_cb(self, msg: VtolVehicleStatus):
        self.vtol_state = msg.vehicle_vtol_state

    def _global_pos_cb(self, msg: VehicleGlobalPosition):
        self.lat      = msg.lat
        self.lon      = msg.lon
        self.alt_amsl = msg.alt
        # Capture ground elevation once while still on the ground
        if self.home_amsl is None and self.state in ("IDLE", "ARMING"):
            self.home_amsl = msg.alt
            self.get_logger().info(f"Home AMSL captured: {self.home_amsl:.1f}m")

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        self.local_alt = -msg.z   # NED: Z is negative-down

    def _attitude_cb(self, msg: VehicleAttitude):
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        self.current_roll, self.current_pitch, self.current_yaw = quaternion_to_euler(q)

    # =========================================================================
    # Threads for MAVLink reading and logging
    # =========================================================================
    def _mav_reader(self):
        """Background thread: receive ATTITUDE_TARGET via pymavlink and cache it thread-safely.

        ATTITUDE_TARGET carries the autopilot's desired attitude setpoint:
          q              — quaternion [w, x, y, z]
          body_roll_rate / body_pitch_rate / body_yaw_rate — desired body rates
          thrust         — collective thrust

        Only this thread calls recv_match, avoiding any pymavlink thread-safety issues.
        The RTA loop and other ROS callbacks read self._desired_* under the lock.
        """
        while rclpy.ok():
            msg = self._mav.recv_match(type='ATTITUDE_TARGET', blocking=True, timeout=1.0)
            if msg is None:
                continue
            # ATTITUDE_TARGET.q is [w, x, y, z]
            roll, pitch, yaw = quaternion_to_euler(
                [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
            )
            with self._att_target_lock:
                self._desired_roll  = roll
                self._desired_pitch = pitch
                self._desired_yaw   = yaw

    def get_desired_attitude(self):
        """Return (roll, pitch, yaw) in radians from the latest ATTITUDE_TARGET.

        Returns (None, None, None) if no message has been received yet.
        Safe to call from any thread.
        """
        with self._att_target_lock:
            return self._desired_roll, self._desired_pitch, self._desired_yaw
        
    # =========================================================================
    # Command helpers
    # =========================================================================

    def _send_cmd(self, command: int, p1=0.0, p2=0.0, p3=0.0,
                  p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(p1)
        msg.param2           = float(p2)
        msg.param3           = float(p3)
        msg.param4           = float(p4)
        msg.param5           = float(p5)
        msg.param6           = float(p6)
        msg.param7           = float(p7)
        msg.target_system    = int(self.model[-1]) + 1 
        msg.target_component = 1
        msg.source_system    = 255
        msg.source_component = 0
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)

        self.cmd_pub.publish(msg)

    def _arm(self):
        self._send_cmd(self.CMD_ARM, p1=1.0)
        self.get_logger().info("ARM sent")

    def _vtol_mc_takeoff(self):
        self._send_cmd(
                self.CMD_SET_MODE,
                p1=float(self.MAV_MODE_FLAG_CUSTOM),
                p2=float(self.MAIN_MODE_AUTO),
                p3=float(self.SUB_MODE_AUTO_TAKEOFF),
            )
        self.get_logger().info("VTOL TAKEOFF sent")
    
    def _vtol_mc_land(self):
        self._send_cmd(
                self.CMD_SET_MODE,
                p1=float(self.MAV_MODE_FLAG_CUSTOM),
                p2=float(self.MAIN_MODE_AUTO),
                p3=float(self.SUB_MODE_AUTO_RTL),
            )
        self.get_logger().info("VTOL LAND sent")
        
    def _vtol_mc_loiter(self):
        self._send_cmd(
                self.CMD_SET_MODE,
                p1=float(self.MAV_MODE_FLAG_CUSTOM),
                p2=float(self.MAIN_MODE_AUTO),
                p3=float(self.SUB_MODE_AUTO_LOITER),
            )
        self.get_logger().info("VTOL LOITER sent")
        
    def _vtol_transition_to_fw(self):
        self._send_cmd(self.CMD_VTOL_TRANS, p1=4.0, p2=0.0)
        self.get_logger().info("VTOL TRANSITION -> fixed-wing sent")

    def _vtol_transition_to_mc(self):
        self._send_cmd(self.CMD_VTOL_TRANS, p1=3.0, p2=0.0)
        self.get_logger().info("VTOL TRANSITION -> multicopter sent")

    def _fly_to_waypoint(self, wp: dict):
        ''' Fly to waypoint using DO_REPOSITION (not ideal for real-time control, but simple for demo purposes) 
            p1: acceptance radius (m) — set to -1 to use default
            p2: 0 = normal reposition, 1 = loiter at waypoint until further notice
            p3: loiter radius (m) if p2=1 (0 to use default)
            p4: yaw at waypoint (NaN to keep current yaw)
            p5-p7: lat, lon, alt (relative to home) of waypoint

            This does conversion from local NED alt to AMSL altitude using the captured home AMSL. 
            This is necessary because PX4's DO_REPOSITION expects absolute AMSL altitude, but our mission
              waypoints are defined in relative AGL terms for simplicity.
        '''
        if self.home_amsl is None:
            self.get_logger().warn("home_amsl not yet captured — cannot send reposition")
            return
        abs_alt = self.home_amsl + wp["alt"]   # convert AGL -> AMSL
        self._send_cmd(
            self.CMD_DO_REPOSITION,
            p1=-1.0,
            p2=0.0,
            p3=0.0,
            p4=float("nan"),
            p5=wp["lat"],
            p6=wp["lon"],
            p7=abs_alt,
        )