"""Inspect the SCARA URDF in RViz with joint sliders.

    ros2 launch scara_description display.launch.py

Use this first -- if the arm looks wrong here, no amount of RL will fix it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("scara_description")
    urdf = os.path.join(share, "urdf", "scara.urdf")
    with open(urdf, "r") as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[{"robot_description": robot_desc}]),
        Node(package="joint_state_publisher_gui",
             executable="joint_state_publisher_gui", output="screen"),
        Node(package="rviz2", executable="rviz2", output="screen"),
    ])
