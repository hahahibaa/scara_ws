"""Fan the trained policy's 4-element joint command out to Gazebo's split
ros2_control controllers.

Subscribes  /scara/joint_command           std_msgs/msg/Float64MultiArray
            (policy output, order [shoulder_pan, elbow_pan, z_lift, wrist_roll])
Publishes   /arm_controller/commands       std_msgs/msg/Float64MultiArray
            (order [shoulder_pan, elbow_pan, wrist_roll] -- matches
            arm_controller's configured joint list, no reindexing needed)
            /z_lift_controller/joint_trajectory   trajectory_msgs/msg/JointTrajectory
            (single-point trajectory -- z_lift is on its own effort-based
            JointTrajectoryController; see scara_description/config/
            scara_controllers.yaml for why)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# time_from_start must be non-zero or JointTrajectoryController rejects the
# point outright.
Z_LIFT_TIME_FROM_START = Duration(sec=0, nanosec=100_000_000)


class PolicyBridge(Node):
    def __init__(self):
        super().__init__("policy_bridge")
        self.pub_arm = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)
        self.pub_zlift = self.create_publisher(JointTrajectory, "/z_lift_controller/joint_trajectory", 10)
        self.create_subscription(Float64MultiArray, "/scara/joint_command", self.on_cmd, 10)
        self.get_logger().info("policy bridge up")

    def on_cmd(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn(f"expected 4 elements, got {len(msg.data)}, ignoring")
            return
        shoulder_pan, elbow_pan, z_lift, wrist_roll = msg.data

        self.pub_arm.publish(Float64MultiArray(data=[shoulder_pan, elbow_pan, wrist_roll]))

        traj = JointTrajectory()
        traj.joint_names = ["z_lift"]
        pt = JointTrajectoryPoint()
        pt.positions = [z_lift]
        pt.time_from_start = Z_LIFT_TIME_FROM_START
        traj.points = [pt]
        self.pub_zlift.publish(traj)


def main():
    rclpy.init()
    node = PolicyBridge()
    try:
        # rclpy.spin(node)'s blocking wait does not reliably deliver
        # subscribed/published data on this host (WSL2 + rmw_fastrtps_cpp);
        # a manually-timed poll loop does.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
