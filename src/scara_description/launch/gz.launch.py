"""SCARA in Gazebo Fortress (headless) + RViz, driven by ros2_control.

    ros2 launch scara_description gz.launch.py

Gazebo's GUI crashes under WSL2 (Ogre/hardware-GL), so this runs the
server only (-s, no window) and visualises through RViz instead -- same
pattern as scara_sim/launch/gz.launch.py, applied to scara_description's
real joint names (shoulder_pan, elbow_pan, z_lift, wrist_roll).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('scara_description')
    urdf = os.path.join(pkg, 'urdf', 'scara.urdf')
    with open(urdf, 'r') as f:
        robot_desc = f.read()
    # This build of gz_ros2_control does not resolve $(find pkg) inside the
    # <parameters> tag itself (it's passed through literally and fails to
    # open as a filename) -- substitute the real path before publishing
    # robot_description instead.
    robot_desc = robot_desc.replace(
        '$(find scara_description)', pkg)
    rviz_cfg = os.path.join(pkg, 'config', 'scara.rviz')
    # empty.sdf (ign-gazebo6's built-in default world) does not load the
    # Sensors system, so camera sensors on any spawned model never render --
    # this is the same world plus that one plugin, loaded by absolute path.
    world = os.path.join(pkg, 'worlds', 'scara_world.sdf')

    # Tell Gazebo where the ROS control plugin lives (fixes "couldn't find
    # shared library"). LIBGL_ALWAYS_SOFTWARE forces Mesa's llvmpipe CPU
    # rasterizer instead of the WSL2 D3D12 hardware-GL translation, which
    # crashed Ogre2's camera-sensor rendering (Ogre::UnimplementedException
    # in GL3PlusTextureGpu::copyTo) -- confirmed necessary, not precautionary.
    env = dict(os.environ, IGN_GAZEBO_SYSTEM_PLUGIN_PATH='/opt/ros/humble/lib',
              LIBGL_ALWAYS_SOFTWARE='1')

    # Pure headless server (-s). No GUI = no Ogre = no WSL crash.
    gz = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', '-v', '2', world],
        output='screen', additional_env=env)

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', 'scara', '-z', '0.0'], output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/overhead_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ], output='screen')
    # rviz2's hardware-GL context and gz's software-rendered camera sensor
    # were found to contend for WSL2's GPU translation layer when both ran
    # concurrently -- with rviz2 on hardware GL, the camera topic degraded
    # from a steady ~8.5Hz to bursty ~5s stalls. Forcing rviz2 onto the same
    # software path removes the contention entirely.
    rviz = Node(package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', rviz_cfg], parameters=[{'use_sim_time': True}],
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'})

    jsb = Node(package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'], output='screen')
    arm = Node(package='controller_manager', executable='spawner',
        arguments=['arm_controller', '--controller-manager-timeout', '60'], output='screen')
    zlift = Node(package='controller_manager', executable='spawner',
        arguments=['z_lift_controller', '--controller-manager-timeout', '60'], output='screen')

    return LaunchDescription([
        gz, rsp, spawn, bridge, rviz,
        RegisterEventHandler(OnProcessExit(target_action=spawn,
            on_exit=[TimerAction(period=3.0, actions=[jsb])])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[zlift])),
    ])
