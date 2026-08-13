"""SCARA (RRPR) geometry, forward and inverse kinematics.

These constants mirror src/scara_description/urdf/scara.urdf exactly. If you
change a length in one place, change it in the other -- everything downstream
(RL env, policy node, vision node) imports from here.
"""

import numpy as np

# ---------------------------------------------------------------- geometry --
COLUMN_H = 0.40   # shoulder axis height
A1 = 0.25         # shoulder_pan -> elbow_pan
A2 = 0.20         # elbow_pan    -> z_lift
QUILL = 0.10      # z_lift joint -> wrist_roll
TOOL = 0.03       # wrist_roll   -> tool tip face

# Tool tip height when the prismatic joint is at d3:  z = TIP_Z0 + d3
TIP_Z0 = COLUMN_H - QUILL - TOOL   # 0.27

JOINT_NAMES = ["shoulder_pan", "elbow_pan", "z_lift", "wrist_roll"]

# Must match the <limit> tags in the URDF.
JOINT_LOWER = np.array([-2.356, -2.530, -0.27, -3.14], dtype=np.float32)
JOINT_UPPER = np.array([2.356, 2.530, 0.00, 3.14], dtype=np.float32)

R_MIN = abs(A1 - A2)   # 0.05
R_MAX = A1 + A2        # 0.45


def forward_kinematics(q):
    """Joint vector [q1, q2, d3, q4] -> (x, y, z, yaw) of the tool tip."""
    q1, q2, d3, q4 = q
    x = A1 * np.cos(q1) + A2 * np.cos(q1 + q2)
    y = A1 * np.sin(q1) + A2 * np.sin(q1 + q2)
    z = TIP_Z0 + d3
    yaw = q1 + q2 + q4
    return np.array([x, y, z, yaw], dtype=np.float64)


def inverse_kinematics(x, y, z, yaw=0.0, elbow_up=True):
    """(x, y, z, yaw) -> joint vector, or None if the pose is unreachable.

    A SCARA has a closed-form IK, so this is exact. It is used as the analytic
    baseline the RL policy is compared against, and to sanity-check the env.
    """
    r2 = x * x + y * y
    r = np.sqrt(r2)
    if r > R_MAX - 1e-6 or r < R_MIN + 1e-6:
        return None

    cos_q2 = (r2 - A1 * A1 - A2 * A2) / (2.0 * A1 * A2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    if not elbow_up:
        q2 = -q2

    q1 = np.arctan2(y, x) - np.arctan2(A2 * np.sin(q2), A1 + A2 * np.cos(q2))
    d3 = z - TIP_Z0
    q4 = yaw - q1 - q2

    # wrap the revolute joints into [-pi, pi]
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    q4 = (q4 + np.pi) % (2 * np.pi) - np.pi

    q = np.array([q1, q2, d3, q4], dtype=np.float64)
    if np.any(q < JOINT_LOWER - 1e-6) or np.any(q > JOINT_UPPER + 1e-6):
        return None
    return q


def in_workspace(x, y, z):
    r = np.hypot(x, y)
    return (R_MIN < r < R_MAX) and (TIP_Z0 + JOINT_LOWER[2] <= z <= TIP_Z0)


# ------------------------------------------------------------------- task --
# Cubes spawn in front of the robot, bins sit off to the left. Both regions are
# inside the annulus and inside the shoulder limit, and they do not overlap.
CUBE_SIZE = 0.04
CUBE_R_RANGE = (0.20, 0.38)
CUBE_THETA_RANGE = (-0.55, 0.55)

COLORS = ["red", "green", "blue"]
COLOR_RGBA = {
    "red":   (0.85, 0.11, 0.13, 1.0),
    "green": (0.13, 0.70, 0.24, 1.0),
    "blue":  (0.12, 0.35, 0.85, 1.0),
}

# Bin centres, one per colour, in the same index order as COLORS.
BIN_RADIUS = 0.30
BIN_ANGLES = [1.30, 1.70, 2.10]
BIN_XY = np.array(
    [[BIN_RADIUS * np.cos(a), BIN_RADIUS * np.sin(a)] for a in BIN_ANGLES],
    dtype=np.float32,
)
BIN_TOL = 0.05   # cube must land within this radius of the bin centre

# Tool tip must get within this 3-D distance of the cube centre to grip it,
# which forces the policy to actually drive the prismatic joint down.
GRASP_RADIUS = 0.035


# --------------------------------------------------------- overhead camera --
# sim_node renders with these; vision_node inverts them. Shared so the two can
# never disagree about where a pixel is in the world.
CAM_EYE_Z = 1.20
CAM_CENTER = (0.10, 0.14)
CAM_FOV = 45.0          # vertical, degrees
CAM_W = 320
CAM_H = 320


def pixel_to_world(u, v, plane_z=CUBE_SIZE, width=CAM_W, height=CAM_H):
    """Un-project a pixel from the top-down camera onto a horizontal plane.

    The camera looks straight down with world +X to image right and world +Y
    to image up, so this is a pure scale about the image centre.

    Un-projecting the centroid of a cube's *top face* at plane_z = CUBE_SIZE
    gives the cube's true xy centre with no perspective error, because the top
    face is centred on the cube.
    """
    h = CAM_EYE_Z - plane_z
    extent = 2.0 * h * np.tan(np.radians(CAM_FOV) / 2.0)   # metres across the view
    scale = extent / height
    x = CAM_CENTER[0] + (u - width / 2.0 + 0.5) * scale
    y = CAM_CENTER[1] - (v - height / 2.0 + 0.5) * scale
    return float(x), float(y)


def self_test():
    """FK(IK(pose)) == pose for a grid of reachable poses."""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(2000):
        r = rng.uniform(R_MIN + 0.02, R_MAX - 0.02)
        th = rng.uniform(-1.5, 1.5)
        x, y = r * np.cos(th), r * np.sin(th)
        z = rng.uniform(TIP_Z0 - 0.27, TIP_Z0)
        q = inverse_kinematics(x, y, z)
        if q is None:
            continue
        got = forward_kinematics(q)
        if not np.allclose(got[:3], [x, y, z], atol=1e-6):
            bad += 1
    return bad


if __name__ == "__main__":
    print(f"reach annulus: {R_MIN:.3f} .. {R_MAX:.3f} m")
    print(f"tip height   : {TIP_Z0 + JOINT_LOWER[2]:.3f} .. {TIP_Z0:.3f} m")
    print(f"bins         : {BIN_XY.tolist()}")
    print(f"FK/IK round-trip mismatches: {self_test()}")
