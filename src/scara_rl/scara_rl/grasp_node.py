"""Automatic magnetic/proximity grip for Gazebo, replicating sim_node.py's
PyBullet grip mechanics exactly (same thresholds, same rigid offset, same
release condition) since the trained policy depends on those semantics.

Subscribes  /tf                    (tool_tip world position)
Publishes   /scara/grasped         std_msgs/msg/Bool
            /scara/grasped_colour  std_msgs/msg/String

grasp_node grabs whichever not-yet-placed cube the tip is actually closest
to (ground truth), which is not necessarily the colour policy_node believes
it is aiming for (that belief comes from vision). /scara/grasped_colour
reports which colour is ACTUALLY held ("" when not holding anything) so
policy_node can resync instead of diverging from reality.

/scara/grasped_colour is the SOLE source of truth for grasp state in
policy_node -- not read alongside /scara/grasped. An earlier version tried
to correlate this topic with the separate Bool topic (sticky colour +
boolean transition), which turned out to still race: there is no ordering
guarantee between two independently-published topics, so the colour string
could still be read one tick stale relative to the boolean flipping true,
resyncing to the PREVIOUS grasp's colour instead of the current one. A
single topic that atomically carries both "holding?" (non-empty) and
"holding what" (the string itself) in one message removes the race by
construction: there is nothing left to correlate. /scara/grasped (Bool) is
still published for any other consumer, but policy_node must not use it
for state transitions.

Grasp trigger  (from scara_env.py / sim_node.py):
    3D distance(tip, cube) < K.GRASP_RADIUS (0.035 m)
Rigid offset while held  (derived from sim_node.py's pybullet constraint:
    createConstraint(..., pad_link, cube, JOINT_FIXED,
                      parentFramePosition=[0,0,-(TOOL+CUBE_SIZE/2)],
                      childFramePosition=[0,0,0])
    => cube_centre = link4_origin - (0,0,TOOL+CUBE_SIZE/2)
       tool_tip    = link4_origin - (0,0,TOOL)
    => cube_centre = tool_tip - (0,0,CUBE_SIZE/2), same XY as the tip):
    cube_centre = tip + (0, 0, -CUBE_SIZE/2)
Release trigger  (from scara_env.py / sim_node.py):
    2D distance(cube_xy, BIN_XY[colour]) < K.BIN_TOL (0.05 m)

Gazebo has no direct equivalent of pybullet's createConstraint reachable
from a plain ROS node here (no gz-transport Python bindings installed on
this host, and each `ign service` CLI call costs ~0.65s of process-start
overhead -- confirmed by direct measurement, not assumption -- so it can't
run in a tight per-tick loop). So the grasp/release DECISION logic runs on
live TF at full rate (correct and instantaneous), while the cube's visual
pose is kept in sync via non-blocking, rate-limited `ign service` calls
that are skipped if a previous one hasn't finished, so they never stall
the decision loop.

Cube starting positions are hardcoded to match
scara_description/worlds/scara_world.sdf's static test cubes -- this is a
fixed-scene verification setup, not the full random-spawn training scene.
"""

import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from . import scara_kinematics as K

CUBE_START = {
    "red":   np.array([0.32, 0.05, K.CUBE_SIZE / 2], dtype=np.float64),
    "green": np.array([0.08, 0.42, K.CUBE_SIZE / 2], dtype=np.float64),
    "blue":  np.array([0.20, -0.05, K.CUBE_SIZE / 2], dtype=np.float64),
}

POSE_SYNC_PERIOD = 0.5   # ign service set_pose costs ~0.65s/call; rate-limited


class GraspNode(Node):
    def __init__(self):
        super().__init__("grasp_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_grasped = self.create_publisher(Bool, "/scara/grasped", 10)
        self.pub_grasped_colour = self.create_publisher(
            String, "/scara/grasped_colour", 10)

        self.cube_pos = {k: v.copy() for k, v in CUBE_START.items()}
        self.grip = None       # colour currently held, or None
        self.placed = set()
        self._sync_proc = None
        self._last_sync = 0.0

        self.get_logger().info("grasp node up")

    def tip_xyz(self):
        try:
            tf = self.tf_buffer.lookup_transform("world", "tool_tip", Time())
        except Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float64)

    def set_cube_pose_async(self, color, pos):
        if self._sync_proc is not None and self._sync_proc.poll() is None:
            return   # previous call still in flight, skip this one
        req = (f'name: "cube_{color}", position: {{x: {pos[0]}, y: {pos[1]}, '
               f'z: {pos[2]}}}, orientation: {{w: 1.0}}')
        self._sync_proc = subprocess.Popen(
            ["ign", "service", "-s", "/world/empty/set_pose",
             "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
             "--timeout", "2000", "--req", req],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tick(self):
        tip = self.tip_xyz()
        if tip is None:
            return

        if self.grip is None:
            for color, pos in self.cube_pos.items():
                if color in self.placed:
                    continue
                if float(np.linalg.norm(tip - pos)) < K.GRASP_RADIUS:
                    self.grip = color
                    self.get_logger().info(f"gripped {color}")
                    break
        else:
            color = self.grip
            new_pos = tip + np.array([0.0, 0.0, -K.CUBE_SIZE / 2])
            self.cube_pos[color] = new_pos

            now = time.monotonic()
            if now - self._last_sync > POSE_SYNC_PERIOD:
                self.set_cube_pose_async(color, new_pos)
                self._last_sync = now

            ci = K.COLORS.index(color)
            d = float(np.linalg.norm(new_pos[:2] - K.BIN_XY[ci]))
            if d < K.BIN_TOL:
                self.get_logger().info(
                    f"released {color} ({len(self.placed) + 1}/3)")
                self.placed.add(color)
                self.grip = None

        self.pub_grasped.publish(Bool(data=self.grip is not None))
        self.pub_grasped_colour.publish(String(data=self.grip or ""))


def main():
    rclpy.init()
    node = GraspNode()
    try:
        # rclpy.spin(node) and create_timer() are both unreliable on this
        # host; a manually-timed poll loop calling tick() directly is what
        # actually works (established and verified earlier this session).
        period = 1.0 / 20.0
        next_tick = time.monotonic()
        while rclpy.ok():
            now = time.monotonic()
            if now >= next_tick:
                node.tick()
                next_tick += period
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
