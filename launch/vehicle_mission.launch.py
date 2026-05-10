from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    vehicle1 = Node(
        package="rta",
        executable="PX4Vehicle1.py",
        name="px4_vehicle1_mission",
        output="screen",
        emulate_tty=True,
    )

    vehicle2 = Node(
        package="rta",
        executable="PX4Vehicle2.py",
        name="px4_vehicle2_mission",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([vehicle1, vehicle2])
