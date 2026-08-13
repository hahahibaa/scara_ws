"""PyBullet gymnasium env: colour-conditioned SCARA pick and place.

Task
----
A cube of a random colour (red / green / blue) spawns in front of the robot.
Three bins sit off to the left, one per colour. The policy must drive the tool
tip down onto the cube, grip it, and carry it to the bin that matches the
cube's colour.

The bin coordinates are deliberately NOT in the observation -- only a colour
one-hot is. So the policy has to learn the colour -> location mapping itself,
which is what makes this an actual colour-sorting policy rather than a
"go to the coordinates you were handed" policy.

Observation (21,)
    q          (4)  joint positions  [q1, q2, d3, q4]
    qdot       (4)  joint velocities
    tip_xyz    (3)  tool tip position
    cube_xyz   (3)  cube position
    tip - cube (3)  error vector
    grasped    (1)  0.0 / 1.0
    colour     (3)  one-hot

Action (4,)  in [-1, 1], scaled to a per-step joint delta, tracked by
PyBullet position control. Position control rather than torque keeps the
policy stable enough to converge within a single training session.

Gripping is automatic on proximity (a "magnetic" suction pad): modelling
finger contact dynamics would triple the training time for no benefit on a
suction-style SCARA.
"""

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from . import scara_kinematics as K

def _find_urdf():
    """Source tree when training, ament share dir when running as a ROS node."""
    src = (Path(__file__).resolve().parents[2]
           / "scara_description" / "urdf" / "scara.urdf")
    if src.exists():
        return src
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("scara_description")) / "urdf" / "scara.urdf"


URDF_PATH = _find_urdf()

# Per-step joint deltas at 30 Hz control -> 1.8 rad/s, 1.8 rad/s, 0.45 m/s, 4.5 rad/s.
# All within the URDF velocity limits.
ACTION_SCALE = np.array([0.060, 0.060, 0.015, 0.150], dtype=np.float32)

CONTROL_HZ = 30
SIM_HZ = 240
SUBSTEPS = SIM_HZ // CONTROL_HZ


class ScaraPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode=None, max_steps=300, fixed_color=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.fixed_color = fixed_color   # pin the colour, for per-colour eval

        self.observation_space = spaces.Box(-np.inf, np.inf, (21,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self.cid)
        if render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            p.resetDebugVisualizerCamera(1.1, 55, -35, [0.1, 0.1, 0.1],
                                         physicsClientId=self.cid)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF(str(URDF_PATH), [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)
        self._map_joints()
        self._make_bin_markers()

        self.cube = None
        self.grip = None
        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------- setup --
    def _map_joints(self):
        """Resolve joint and link indices by name -- never hard-code them."""
        self.jidx, self.lidx = {}, {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            self.jidx[info[1].decode()] = i
            self.lidx[info[12].decode()] = i
        self.arm = [self.jidx[n] for n in K.JOINT_NAMES]
        self.tip_link = self.lidx["tool_tip"]
        self.pad_link = self.lidx["link4"]

    def _make_bin_markers(self):
        """Bins are visual-only discs. Real walls would add collisions the
        policy has to fight through, which is not what we are training."""
        self.bin_ids = []
        for i, name in enumerate(K.COLORS):
            rgba = list(K.COLOR_RGBA[name])
            rgba[3] = 0.45
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=K.BIN_TOL,
                                     length=0.004, rgbaColor=rgba,
                                     physicsClientId=self.cid)
            bid = p.createMultiBody(0, -1, vs,
                                    [K.BIN_XY[i][0], K.BIN_XY[i][1], 0.002],
                                    physicsClientId=self.cid)
            self.bin_ids.append(bid)

    def _spawn_cube(self):
        if self.cube is not None:
            p.removeBody(self.cube, physicsClientId=self.cid)
        h = K.CUBE_SIZE / 2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h],
                                     physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h, h, h],
                                  rgbaColor=K.COLOR_RGBA[K.COLORS[self.color_idx]],
                                  physicsClientId=self.cid)
        self.cube = p.createMultiBody(0.05, col, vis,
                                      [self.cube_xy0[0], self.cube_xy0[1], h],
                                      physicsClientId=self.cid)
        p.changeDynamics(self.cube, -1, lateralFriction=0.9, spinningFriction=0.005,
                         physicsClientId=self.cid)

    # -------------------------------------------------------------- state --
    def _joint_state(self):
        st = p.getJointStates(self.robot, self.arm, physicsClientId=self.cid)
        q = np.array([s[0] for s in st], dtype=np.float32)
        qd = np.array([s[1] for s in st], dtype=np.float32)
        return q, qd

    def _tip_xyz(self):
        ls = p.getLinkState(self.robot, self.tip_link, computeForwardKinematics=True,
                            physicsClientId=self.cid)
        return np.array(ls[0], dtype=np.float32)

    def _cube_xyz(self):
        pos, _ = p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)
        return np.array(pos, dtype=np.float32)

    def _obs(self):
        q, qd = self._joint_state()
        tip = self._tip_xyz()
        cube = self._cube_xyz()
        onehot = np.zeros(3, dtype=np.float32)
        onehot[self.color_idx] = 1.0
        return np.concatenate([
            q, qd, tip, cube, tip - cube,
            np.array([1.0 if self.grip is not None else 0.0], dtype=np.float32),
            onehot,
        ]).astype(np.float32)

    # -------------------------------------------------------------- gym --
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        rng = self._np_random

        self.color_idx = (self.fixed_color if self.fixed_color is not None
                          else int(rng.integers(0, 3)))
        r = rng.uniform(*K.CUBE_R_RANGE)
        th = rng.uniform(*K.CUBE_THETA_RANGE)
        self.cube_xy0 = np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float32)

        if self.grip is not None:
            p.removeConstraint(self.grip, physicsClientId=self.cid)
            self.grip = None

        # home pose, arm folded out of the way and quill retracted
        for j, val in zip(self.arm, [0.0, 0.6, 0.0, 0.0]):
            p.resetJointState(self.robot, j, val, 0.0, physicsClientId=self.cid)
        self.target = np.array([0.0, 0.6, 0.0, 0.0], dtype=np.float32)

        self._spawn_cube()
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.cid)

        self.steps = 0
        self.prev_reach = float(np.linalg.norm(self._tip_xyz() - self._cube_xyz()))
        self.prev_carry = None
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self.target = np.clip(self.target + action * ACTION_SCALE,
                              K.JOINT_LOWER, K.JOINT_UPPER)
        p.setJointMotorControlArray(
            self.robot, self.arm, p.POSITION_CONTROL,
            targetPositions=self.target.tolist(),
            forces=[60.0, 40.0, 80.0, 10.0],
            physicsClientId=self.cid,
        )
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        self.steps += 1
        tip, cube = self._tip_xyz(), self._cube_xyz()

        reward = -0.02                       # time cost
        reward -= 0.005 * float(action @ action)
        terminated = False
        info = {"color": K.COLORS[self.color_idx], "is_success": False}

        if self.grip is None:
            # ---- phase 1: reach and grip -------------------------------
            d = float(np.linalg.norm(tip - cube))
            reward += 10.0 * (self.prev_reach - d)     # potential shaping
            self.prev_reach = d
            if d < K.GRASP_RADIUS:
                self.grip = p.createConstraint(
                    self.robot, self.pad_link, self.cube, -1, p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, -(K.TOOL + K.CUBE_SIZE / 2)], [0, 0, 0],
                    physicsClientId=self.cid,
                )
                p.changeConstraint(self.grip, maxForce=200, physicsClientId=self.cid)
                reward += 5.0
                self.prev_carry = float(
                    np.linalg.norm(cube[:2] - K.BIN_XY[self.color_idx])
                )
        else:
            # ---- phase 2: carry to the colour-matched bin --------------
            d = float(np.linalg.norm(cube[:2] - K.BIN_XY[self.color_idx]))
            reward += 10.0 * (self.prev_carry - d)
            self.prev_carry = d
            if d < K.BIN_TOL:
                p.removeConstraint(self.grip, physicsClientId=self.cid)
                self.grip = None
                reward += 20.0
                terminated = True
                info["is_success"] = True
            elif cube[2] < 0.5 * K.CUBE_SIZE:
                # cube slipped off the pad
                p.removeConstraint(self.grip, physicsClientId=self.cid)
                self.grip = None
                reward -= 5.0
                terminated = True

        # cube shoved outside the reachable annulus -> unrecoverable
        if not terminated:
            rr = float(np.hypot(cube[0], cube[1]))
            if rr > K.R_MAX - 0.01 or rr < K.R_MIN + 0.01:
                reward -= 5.0
                terminated = True

        truncated = self.steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------ camera --
    def render_overhead(self, width=320, height=320, eye_z=1.20,
                        center=(0.10, 0.14), fov=45.0):
        """Synthetic top-down RGB frame -- this is what the vision node sees."""
        eye = [center[0], center[1], eye_z]
        tgt = [center[0], center[1], 0.0]
        view = p.computeViewMatrix(eye, tgt, [0, 1, 0], physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(fov, width / height, 0.1, 3.0,
                                            physicsClientId=self.cid)
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.cid,
        )
        return np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(physicsClientId=self.cid)
