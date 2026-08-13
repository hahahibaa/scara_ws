#!/usr/bin/env bash
set -e
PKG=~/scara_ws/src/scara_sim
mkdir -p "$PKG"/urdf "$PKG"/config "$PKG"/launch

cat > "$PKG"/package.xml << 'EOF'
<?xml version="1.0"?>
<package format="3">
  <name>scara_sim</name>
  <version>0.0.1</version>
  <description>SCARA arm sim in Gazebo Fortress with ros2_control</description>
  <maintainer email="hiba@example.com">hiba</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>ros_gz_sim</exec_depend>
  <exec_depend>ros_gz_bridge</exec_depend>
  <exec_depend>controller_manager</exec_depend>
  <exec_depend>joint_state_broadcaster</exec_depend>
  <exec_depend>position_controllers</exec_depend>
  <exec_depend>xacro</exec_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
EOF

cat > "$PKG"/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.8)
project(scara_sim)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf config launch DESTINATION share/${PROJECT_NAME})
ament_package()
EOF

cat > "$PKG"/urdf/scara.urdf.xacro << 'EOF'
<?xml version="1.0"?>
<robot name="scara" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <material name="grey">   <color rgba="0.4 0.4 0.4 1"/></material>
  <material name="blue">   <color rgba="0.2 0.4 0.8 1"/></material>
  <material name="orange"> <color rgba="0.9 0.5 0.1 1"/></material>

  <link name="base_link">
    <visual><origin xyz="0 0 0.2"/><geometry><cylinder radius="0.06" length="0.4"/></geometry><material name="grey"/></visual>
    <collision><origin xyz="0 0 0.2"/><geometry><cylinder radius="0.06" length="0.4"/></geometry></collision>
    <inertial><origin xyz="0 0 0.2"/><mass value="2.0"/><inertia ixx="0.03" iyy="0.03" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
  </joint>
  <link name="link1">
    <visual><origin xyz="0.15 0 0"/><geometry><box size="0.3 0.06 0.06"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="0.15 0 0"/><geometry><box size="0.3 0.06 0.06"/></geometry></collision>
    <inertial><origin xyz="0.15 0 0"/><mass value="1.0"/><inertia ixx="0.005" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2.5" upper="2.5" effort="100" velocity="2.0"/>
  </joint>
  <link name="link2">
    <visual><origin xyz="0.125 0 0"/><geometry><box size="0.25 0.05 0.05"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="0.125 0 0"/><geometry><box size="0.25 0.05 0.05"/></geometry></collision>
    <inertial><origin xyz="0.125 0 0"/><mass value="0.8"/><inertia ixx="0.003" iyy="0.006" izz="0.006" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="joint3" type="prismatic">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.15" upper="0.0" effort="100" velocity="0.5"/>
  </joint>
  <link name="link3">
    <visual><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.02" length="0.2"/></geometry><material name="grey"/></visual>
    <collision><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.02" length="0.2"/></geometry></collision>
    <inertial><origin xyz="0 0 -0.1"/><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/>
    <origin xyz="0 0 -0.2"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="2.0"/>
  </joint>
  <link name="link4">
    <visual><origin xyz="0 0 -0.015"/><geometry><box size="0.06 0.06 0.03"/></geometry><material name="orange"/></visual>
    <collision><origin xyz="0 0 -0.015"/><geometry><box size="0.06 0.06 0.03"/></geometry></collision>
    <inertial><origin xyz="0 0 -0.015"/><mass value="0.1"/><inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <ros2_control name="GazeboSimSystem" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="joint1"><command_interface name="position"/><state_interface name="position"/><state_interface name="velocity"/></joint>
    <joint name="joint2"><command_interface name="position"/><state_interface name="position"/><state_interface name="velocity"/></joint>
    <joint name="joint3"><command_interface name="position"/><state_interface name="position"/><state_interface name="velocity"/></joint>
    <joint name="joint4"><command_interface name="position"/><state_interface name="position"/><state_interface name="velocity"/></joint>
  </ros2_control>

  <gazebo>
    <plugin filename="libgz_ros2_control-system.so" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>$(find scara_sim)/config/scara_controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
EOF

cat > "$PKG"/config/scara_controllers.yaml << 'EOF'
controller_manager:
  ros__parameters:
    update_rate: 100
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    arm_controller:
      type: position_controllers/JointGroupPositionController

arm_controller:
  ros__parameters:
    joints: [joint1, joint2, joint3, joint4]
    interface_name: position
EOF

cat > "$PKG"/launch/gz.launch.py << 'EOF'
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('scara_sim')
    robot_desc = xacro.process_file(os.path.join(pkg,'urdf','scara.urdf.xacro')).toxml()
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim,'launch','gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r empty.sdf'}.items())
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create',
        arguments=['-topic','robot_description','-name','scara','-z','0.0'], output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'], output='screen')
    jsb = Node(package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster'], output='screen')
    arm = Node(package='controller_manager', executable='spawner',
        arguments=['arm_controller'], output='screen')

    return LaunchDescription([gz, rsp, spawn, bridge,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm]))])
EOF
echo ">> scara_sim package created."