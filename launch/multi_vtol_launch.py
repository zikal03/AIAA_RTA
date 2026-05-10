import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── SET YOUR PX4 PATH HERE ─────────────────────────────────────
#PX4_DIR = os.environ.get("PX4_DIR", os.path.expanduser("~/PX4-Autopilot"))
PX4_DIR = "~/PX4-Autopilot"
# ──────────────────────────────────────────────────────────────


def make_px4_process(instance: int, x: float, y: float):
    """
    Launch one PX4 SITL instance.
    PX4_SYS_AUTOSTART=4004 → standard_vtol airframe
    PX4_GZ_MODEL_POSE     → spawn position x,y,z,roll,pitch,yaw
    PX4_SIM_MODEL          → Gazebo model name to use
    Instance 1 starts Gazebo; instances > 1 join the running sim.
    """
    if instance == 1:
        print("Launching PX4 instance 1 (starts Gazebo)")
        env = {
            "PX4_SYS_AUTOSTART": "4004",
            "PX4_GZ_WORLD": "baylands",
            "PX4_SIM_MODEL": "gz_standard_vtol"
        }
    else:
        print(f"Launching PX4 instance {instance} (joins existing Gazebo session)")
        env = {
            "PX4_SYS_AUTOSTART": "4004",
            "PX4_GZ_WORLD": "baylands",
            "PX4_SIM_MODEL": "gz_standard_vtol",
            "PX4_GZ_MODEL_POSE": f"{x},{y},0.3,0,0,0",
            # Tell PX4 which Gazebo server to connect to (all share one gz server)
            "PX4_GZ_STANDALONE": "1",
        }

    cmd = (
        f"cd {PX4_DIR} && "
        f"./build/px4_sitl_default/bin/px4 "
        f"-i {instance} "
    )

    return ExecuteProcess(
        cmd=["bash", "-c", cmd],
        additional_env=env,
        output="screen",
        name=f"px4_instance_{instance}",
    )


def generate_launch_description():

    # ── Drone 0 — spawns at origin, also starts Gazebo ──────────
    drone0_px4   = make_px4_process(instance=1, x=0.0, y=0.0)

    # ── Drone 1 — spawns 5 m away, joins existing Gazebo session ─
    drone1_px4   = TimerAction(period=30.0, actions=[make_px4_process(instance=2, x=5.0, y=10.0)])

    xrce_agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
        output="screen",
        name="xrce_agent",
    )

    agent = TimerAction(period=40.0, actions=[xrce_agent])

    return LaunchDescription([
        drone0_px4,
        drone1_px4,
        agent,
    ])