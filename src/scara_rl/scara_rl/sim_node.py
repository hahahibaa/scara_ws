"""PyBullet simulator exposed on the ROS 2 graph.

Publishes
    /joint_states                 sensor_msgs/JointState
    /overhead_camera/image_raw    sensor_msgs/Image   (top-down RGB)
    /scara/grasped                std_msgs/Bool
Subscribes
    /scara/joint_command          std_msgs/Float64MultiArray  (4 target positions)

Grip and release are automatic and use EXACTLY the thresholds and constraint
parameters from scara_env.py -- if these drift apart the trained policy will
behave differently here than it did in training.
"""

import time

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64MultiArray

from . import scara_kinematics as K
from .scara_env import URDF_PATH, SIM_HZ, SUBSTEPS, CONTROL_HZ

CAM_EYE_Z = 1.20
CAM_CENTER = (0.10, 0.14)
CAM_FOV = 45.0
CAM_W = CAM_H = 320


class ScaraSimNode(Node):
    def __init__(self):
        super().__init__("scara_sim")

        # GUI mode's rendering thread starves the ROS executor under WSL's
        # D3D12/Mesa translation (confirmed: control-loop timers drop from
        # ~20Hz to well under 1Hz with p.GUI connected) and segfaults on
        # exit, so this runs headless. Watch the arm via
        # /overhead_camera/image_raw or /vision/debug_image instead.
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self.cid)

        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF(str(URDF_PATH), [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)

        self.jidx, self.lidx = {}, {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            self.jidx[info[1].decode()] = i
            self.lidx[info[12].decode()] = i
        self.arm = [self.jidx[n] for n in K.JOINT_NAMES]
        self.tip_link = self.lidx["tool_tip"]
        self.pad_link = self.lidx["link4"]

        self._make_bins()
        self._spawn_cubes()

        self.target = np.array([0.0, 0.6, 0.0, 0.0], dtype=np.float64)
        for j, v in zip(self.arm, self.target):
            p.resetJointState(self.robot, j, v, 0.0, physicsClientId=self.cid)

        self.grip = None
        self.grip_cube = None
        self.placed = set()
        self.bridge = CvBridge()

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_img = self.create_publisher(Image, "/overhead_camera/image_raw", 2)
        self.pub_grasp = self.create_publisher(Bool, "/scara/grasped", 10)
        self.create_subscription(Float64MultiArray, "/scara/joint_command",
                                 self.on_cmd, 10)

        # rclpy's create_timer() does not fire reliably on this host (verified:
        # a create_timer-driven 30Hz publisher delivered ~0.7Hz; a manually-timed
        # loop driving the same callback delivered a clean 30Hz), so tick() and
        # publish_image() are called directly from main()'s poll loop instead.
        self.tick_period = 1.0 / CONTROL_HZ
        self.image_period = 1.0 / 5.0
        self.get_logger().info("SCARA sim up. 3 cubes spawned, awaiting commands.")

    # ------------------------------------------------------------- scene --
    def _make_bins(self):
        for i, name in enumerate(K.COLORS):
            rgba = list(K.COLOR_RGBA[name]); rgba[3] = 0.45
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=K.BIN_TOL, length=0.004,
                                     rgbaColor=rgba, physicsClientId=self.cid)
            p.createMultiBody(0, -1, vs, [K.BIN_XY[i][0], K.BIN_XY[i][1], 0.002],
                              physicsClientId=self.cid)

    def _spawn_cubes(self):
        """One cube per colour, spread across the pick region."""
        self.cubes = {}
        h = K.CUBE_SIZE / 2
        rng = np.random.default_rng(7)
        for i, name in enumerate(K.COLORS):
            r = rng.uniform(0.24, 0.34)
            th = -0.45 + i * 0.45
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h],
                                         physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h, h, h],
                                      rgbaColor=K.COLOR_RGBA[name],
                                      physicsClientId=self.cid)
            bid = p.createMultiBody(0.05, col, vis,
                                    [r * np.cos(th), r * np.sin(th), h],
                                    physicsClientId=self.cid)
            p.changeDynamics(bid, -1, lateralFriction=0.9, spinningFriction=0.005,
                             physicsClientId=self.cid)
            self.cubes[name] = bid

    # ------------------------------------------------------------- loop --
    def on_cmd(self, msg):
        if len(msg.data) == 4:
            self.target = np.clip(np.array(msg.data, dtype=np.float64),
                                  K.JOINT_LOWER, K.JOINT_UPPER)

    def tip_xyz(self):
        ls = p.getLinkState(self.robot, self.tip_link, computeForwardKinematics=True,
                            physicsClientId=self.cid)
        return np.array(ls[0])

    def tick(self):
        p.setJointMotorControlArray(
            self.robot, self.arm, p.POSITION_CONTROL,
            targetPositions=self.target.tolist(),
            forces=[60.0, 40.0, 80.0, 10.0], physicsClientId=self.cid,
        )
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        self._update_grip()

        st = p.getJointStates(self.robot, self.arm, physicsClientId=self.cid)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(K.JOINT_NAMES)
        js.position = [s[0] for s in st]
        js.velocity = [s[1] for s in st]
        self.pub_js.publish(js)
        self.pub_grasp.publish(Bool(data=self.grip is not None))

    def _update_grip(self):
        tip = self.tip_xyz()
        if self.grip is None:
            for name, bid in self.cubes.items():
                if name in self.placed:
                    continue
                pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=self.cid)
                if np.linalg.norm(tip - np.array(pos)) < K.GRASP_RADIUS:
                    self.grip = p.createConstraint(
                        self.robot, self.pad_link, bid, -1, p.JOINT_FIXED,
                        [0, 0, 0], [0, 0, -(K.TOOL + K.CUBE_SIZE / 2)], [0, 0, 0],
                        physicsClientId=self.cid)
                    p.changeConstraint(self.grip, maxForce=200,
                                       physicsClientId=self.cid)
                    self.grip_cube = name
                    self.get_logger().info(f"gripped {name}")
                    break
        else:
            name = self.grip_cube
            ci = K.COLORS.index(name)
            pos, _ = p.getBasePositionAndOrientation(self.cubes[name],
                                                     physicsClientId=self.cid)
            if np.linalg.norm(np.array(pos[:2]) - K.BIN_XY[ci]) < K.BIN_TOL:
                p.removeConstraint(self.grip, physicsClientId=self.cid)
                self.grip = None
                self.placed.add(name)
                self.get_logger().info(
                    f"placed {name} -> {name} bin  ({len(self.placed)}/3 done)")
                self.grip_cube = None

    def publish_image(self):
        eye = [CAM_CENTER[0], CAM_CENTER[1], CAM_EYE_Z]
        tgt = [CAM_CENTER[0], CAM_CENTER[1], 0.0]
        view = p.computeViewMatrix(eye, tgt, [0, 1, 0], physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(CAM_FOV, 1.0, 0.1, 3.0,
                                            physicsClientId=self.cid)
        _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=self.cid)
        arr = np.reshape(np.array(rgb, dtype=np.uint8), (CAM_H, CAM_W, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "overhead_camera"
        self.pub_img.publish(msg)


def main():
    rclpy.init()
    node = ScaraSimNode()
    try:
        # rclpy.spin(node)'s blocking wait does not reliably deliver
        # subscribed/published data on this host (WSL2 + rmw_fastrtps_cpp);
        # a manually-timed poll loop does. create_timer() is similarly
        # unreliable here, so tick()/publish_image() are driven directly
        # off elapsed monotonic time instead of rclpy Timers.
        next_tick = time.monotonic()
        next_image = next_tick
        while rclpy.ok():
            now = time.monotonic()
            if now >= next_tick:
                node.tick()
                next_tick += node.tick_period
            if now >= next_image:
                node.publish_image()
                next_image += node.image_period
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
