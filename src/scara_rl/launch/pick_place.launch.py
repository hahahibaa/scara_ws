"""Full demo: PyBullet sim + colour detection + trained policy.

    ros2 launch scara_rl pick_place.launch.py \
        model_path:=$HOME/scara_ws/models/scara_ppo_final.zip
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_model = os.path.join(
        os.path.expanduser("~"), "scara_ws", "models", "scara_ppo_final.zip"
    )

    model_path = LaunchConfiguration("model_path")
    use_vision = LaunchConfiguration("use_vision")

    return LaunchDescription([
        DeclareLaunchArgument("model_path", default_value=default_model),
        DeclareLaunchArgument(
            "use_vision", default_value="true",
            description="run the HSV detector; set false to debug the sim alone"),

        Node(package="scara_rl", executable="sim_node", name="scara_sim",
             output="screen"),

        Node(package="scara_rl", executable="vision_node", name="scara_vision",
             output="screen", condition=IfCondition(use_vision)),

        Node(package="scara_rl", executable="policy_node", name="scara_policy",
             output="screen",
             parameters=[{"model_path": model_path, "deterministic": True}]),
    ])
