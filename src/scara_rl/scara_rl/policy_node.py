"""Run the trained PPO policy against the live ROS 2 graph.

Subscribes  /joint_states       sensor_msgs/JointState
            /detected_objects   scara_msgs/DetectedObjectArray   (from vision)
            /scara/grasped      std_msgs/Bool
Publishes   /scara/joint_command  std_msgs/Float64MultiArray

The observation vector assembled here must be byte-for-byte the same layout as
ScaraPickPlaceEnv._obs(), otherwise the policy is reading garbage. The one
substitution is the cube position: in training it came from the simulator's
ground truth, here it comes from the colour detector. Once the cube is gripped
the detector cannot see it (it is underneath the tool), so we fall back to the
known rigid offset from the tool tip.
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from stable_baselines3 import PPO

from scara_msgs.msg import DetectedObjectArray

from . import scara_kinematics as K
from .scara_env import ACTION_SCALE, CONTROL_HZ


class PolicyNode(Node):
    def __init__(self):
        super().__init__("scara_policy")
        self.declare_parameter("model_path", "")
        self.declare_parameter("deterministic", True)
        path = self.get_parameter("model_path").value
        if not path:
            raise RuntimeError("set the model_path parameter to your .zip")
        self.model = PPO.load(path, device="cpu")
        self.deterministic = bool(self.get_parameter("deterministic").value)
        self.get_logger().info(f"loaded policy: {path}")

        self.q = None
        self.qd = np.zeros(4)
        self.target = None
        self.grasped = False
        self.detections = []
        self.active = None          # colour we are currently working on
        self.done = set()

        self.pub = self.create_publisher(Float64MultiArray,
                                         "/scara/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(DetectedObjectArray, "/detected_objects",
                                 self.on_det, 10)
        self.create_subscription(Bool, "/scara/grasped", self.on_grasp, 10)
        # create_timer() does not fire reliably on this host; tick() is
        # driven directly off elapsed monotonic time from main() instead.
        self.tick_period = 1.0 / CONTROL_HZ

    # ------------------------------------------------------------ inputs --
    def on_js(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [idx[n] for n in K.JOINT_NAMES]
        except KeyError:
            return
        self.q = np.array([msg.position[i] for i in order], dtype=np.float32)
        if msg.velocity:
            self.qd = np.array([msg.velocity[i] for i in order], dtype=np.float32)
        if self.target is None:
            self.target = self.q.astype(np.float64).copy()

    def on_det(self, msg):
        self.detections = [(o.color, np.array([o.position.x, o.position.y,
                                               o.position.z], np.float32))
                           for o in msg.objects]

    def on_grasp(self, msg):
        was = self.grasped
        self.grasped = bool(msg.data)
        if was and not self.grasped and self.active is not None:
            # released -- the sim only releases over the correct bin
            self.done.add(self.active)
            self.get_logger().info(
                f"{self.active} delivered ({len(self.done)}/3)")
            self.active = None

    # ------------------------------------------------------------- logic --
    def pick_target(self, tip):
        """Nearest cube we have not delivered yet."""
        cands = [(c, pos) for c, pos in self.detections if c not in self.done]
        if not cands:
            return None
        return min(cands, key=lambda cp: np.linalg.norm(cp[1] - tip))

    def tick(self):
        if self.q is None or self.target is None:
            return

        tip = K.forward_kinematics(self.q)[:3].astype(np.float32)

        if self.grasped:
            if self.active is None:
                return                       # gripped something unexpected
            color = self.active
            cube = tip + np.array([0.0, 0.0, -K.CUBE_SIZE / 2.0], np.float32)
        else:
            sel = self.pick_target(tip)
            if sel is None:
                if len(self.done) >= 3:
                    self.get_logger().info("all cubes sorted", once=True)
                return
            color, cube = sel
            self.active = color

        onehot = np.zeros(3, dtype=np.float32)
        onehot[K.COLORS.index(color)] = 1.0
        obs = np.concatenate([
            self.q, self.qd, tip, cube, tip - cube,
            np.array([1.0 if self.grasped else 0.0], np.float32),
            onehot,
        ]).astype(np.float32)

        act, _ = self.model.predict(obs, deterministic=self.deterministic)
        act = np.clip(act, -1.0, 1.0)
        self.target = np.clip(self.target + act * ACTION_SCALE,
                              K.JOINT_LOWER, K.JOINT_UPPER)

        self.pub.publish(Float64MultiArray(data=self.target.tolist()))


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        # rclpy.spin(node)'s blocking wait does not reliably deliver
        # subscribed/published data on this host (WSL2 + rmw_fastrtps_cpp);
        # a manually-timed poll loop does. create_timer() is similarly
        # unreliable here, so tick() is driven off elapsed monotonic time.
        next_tick = time.monotonic()
        while rclpy.ok():
            if time.monotonic() >= next_tick:
                node.tick()
                next_tick += node.tick_period
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
