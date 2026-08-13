import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('scara_sim')
    robot_desc = xacro.process_file(os.path.join(pkg, 'urdf', 'scara.urdf.xacro')).toxml()
    rviz_cfg = os.path.join(pkg, 'config', 'scara.rviz')

    # Tell Gazebo where the ROS control plugin lives (fixes "couldn't find shared library")
    env = dict(os.environ, IGN_GAZEBO_SYSTEM_PLUGIN_PATH='/opt/ros/humble/lib')

    # Pure headless server (-s). No GUI = no Ogre = no WSL crash.
    gz = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', '-v', '2', 'empty.sdf'],
        output='screen', additional_env=env)

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', 'scara', '-z', '0.0'], output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'], output='screen')
    rviz = Node(package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', rviz_cfg], parameters=[{'use_sim_time': True}])

    jsb = Node(package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'], output='screen')
    arm = Node(package='controller_manager', executable='spawner',
        arguments=['arm_controller', '--controller-manager-timeout', '60'], output='screen')

    return LaunchDescription([
        gz, rsp, spawn, bridge, rviz,
        RegisterEventHandler(OnProcessExit(target_action=spawn,
            on_exit=[TimerAction(period=3.0, actions=[jsb])])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
    ])