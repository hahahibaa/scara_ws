"""Run the trained PPO policy against the live ROS 2 graph.

Subscribes  /joint_states          sensor_msgs/JointState
            /detected_objects      scara_msgs/DetectedObjectArray   (from vision)
            /scara/grasped_colour  std_msgs/String   (ground truth, from sim_node.py
                                                        or grasp_node.py)
Publishes   /scara/joint_command  std_msgs/Float64MultiArray

The observation vector assembled here must be byte-for-byte the same layout as
ScaraPickPlaceEnv._obs(), otherwise the policy is reading garbage. The one
substitution is the cube position: in training it came from the simulator's
ground truth, here it comes from the colour detector. Once the cube is gripped
the detector cannot see it (it is underneath the tool), so we fall back to the
known rigid offset from the tool tip.

self.active (which colour we believe we're working on) is normally set from
vision when we start reaching for a cube. But the actual grip is granted by
ground-truth proximity (nearest not-yet-placed cube in range -- see
sim_node.py / grasp_node.py), which is not guaranteed to be the same colour
we were aiming for. If belief and reality are allowed to diverge, "done"
gets marked for the wrong colour: the truly-delivered cube then sits inside
its own bin where vision filters it out (bin-proximity rejection), the
colour we're actually still missing is never retargeted, and the whole run
stalls. So self.active is resynced to /scara/grasped_colour's ground truth
the moment a grasp is detected, not just when tallying "done" at release --
that keeps the one-hot observation correct for the entire carry phase too,
not only the final bookkeeping.

grasp state (self.grasped, and the grasp/release transition) is derived
SOLELY from /scara/grasped_colour ("" = not holding, else the colour held)
-- deliberately not cross-checked against the separate /scara/grasped Bool
topic. An earlier version tried to correlate the two (a sticky colour
string + a separate boolean edge), which still raced: there is no ordering
guarantee between independently-published topics, so the colour could be
read one tick stale relative to the boolean flipping, resyncing to the
PREVIOUS grasp's colour. A single topic that atomically carries both
"holding?" and "holding what" in one message removes that race by
construction -- there is nothing left to correlate.
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
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
        self.active = None          # colour we believe we are working on
        self.done = set()

        self.pub = self.create_publisher(Float64MultiArray,
                                         "/scara/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(DetectedObjectArray, "/detected_objects",
                                 self.on_det, 10)
        self.create_subscription(String, "/scara/grasped_colour",
                                 self.on_grasp_colour, 10)
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

    def on_grasp_colour(self, msg):
        colour = msg.data or None   # "" -> None (not holding anything)
        was = self.grasped
        self.grasped = colour is not None

        if not was and self.grasped:
            # just grasped -- ground-truth proximity picks whichever
            # not-yet-placed cube the tip is actually nearest to, which is
            # not guaranteed to be the colour we were aiming for. Resync
            # belief to reality now (from THIS message alone, not a
            # separately-tracked value) so the one-hot observation is
            # correct for the whole carry phase, not just at release.
            if colour != self.active:
                self.get_logger().warn(
                    f"aimed for {self.active}, actually grasped "
                    f"{colour} -- resyncing")
                self.active = colour
        elif was and not self.grasped and self.active is not None:
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
