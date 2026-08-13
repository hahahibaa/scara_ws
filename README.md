# SCARA colour-sorting pick and place — ROS 2 Humble + PyBullet + PPO

A 4-DOF SCARA (RRPR) learns, with PPO, to pick up a coloured cube and drop it
in the bin matching its colour. Training runs in PyBullet; the finished policy
is deployed as a ROS 2 node driving a PyBullet-backed simulator, with an
OpenCV HSV node supplying the object detections.

## Why PyBullet and not Gazebo for training

Gazebo runs at roughly real time. PPO needs on the order of a million
environment steps, which is about 9 hours of simulated time — days of
wall-clock in Gazebo. PyBullet in `DIRECT` mode with 8 parallel workers covers
the same steps in about 15 minutes. Gazebo remains installed and is a fine
visualisation target, but it is not where the learning happens.

## Layout

```
scara_ws/
├── setup.sh                       one-shot dependency install
├── models/                        trained policies land here
├── runs/                          tensorboard logs
└── src/
    ├── scara_description/         URDF + RViz launch
    ├── scara_msgs/                DetectedObject[Array]
    └── scara_rl/scara_rl/
        ├── scara_kinematics.py    geometry, FK, closed-form IK, camera model
        ├── scara_env.py           gymnasium env  (training)
        ├── train_ppo.py           PPO training
        ├── eval_policy.py         success rate, per colour
        ├── sim_node.py            PyBullet on the ROS graph
        ├── vision_node.py         HSV colour detection
        └── policy_node.py         runs the policy against live topics
```

`scara_kinematics.py` is the single source of truth for lengths, joint limits,
bin positions and the camera model. The URDF mirrors it in comments. Change
one, change the other.

## Run order

### 1. Install (once, ~10 min)

```bash
bash ~/scara_ws/setup.sh
```

### 2. Check the geometry

```bash
python3 ~/scara_ws/src/scara_rl/scara_rl/scara_kinematics.py
```

Expect `FK/IK round-trip mismatches: 0`.

### 3. Train

```bash
source ~/scara_venv/bin/activate
cd ~/scara_ws/src/scara_rl
python -m scara_rl.train_ppo --timesteps 1500000
```

Watch `task/success_rate` in `tensorboard --logdir ~/scara_ws/runs`. The reach
phase is learned early; the colour→bin mapping takes longer because it has to
be discovered by exploration.

### 4. Evaluate

```bash
python -m scara_rl.eval_policy --episodes 90            # numbers, per colour
python -m scara_rl.eval_policy --render --episodes 10   # watch it
```

### 5. Build and run the ROS 2 system

```bash
source /opt/ros/humble/setup.bash
cd ~/scara_ws && colcon build --symlink-install
source ~/scara_ws/install/setup.bash
source ~/scara_venv/bin/activate

ros2 launch scara_rl pick_place.launch.py
```

Inspect the pipeline live:

```bash
ros2 topic echo /detected_objects
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

### URDF sanity check in RViz

```bash
ros2 launch scara_description display.launch.py
```

## Robot

| joint          | type      | axis | range              |
|----------------|-----------|------|--------------------|
| `shoulder_pan` | revolute  | Z    | ±135°              |
| `elbow_pan`    | revolute  | Z    | ±145°              |
| `z_lift`       | prismatic | Z    | −0.27 … 0 m        |
| `wrist_roll`   | revolute  | Z    | ±180°              |

Planar reach 0.05–0.45 m; tool tip height 0.00–0.27 m.

## Notes

- A SCARA's IK is closed-form and exact (`inverse_kinematics()` in
  `scara_kinematics.py`). The RL policy can only approximate it. The analytic
  solver is kept as the baseline to compare the policy against.
- Bins are visual-only discs. Solid bin walls would add collisions the policy
  has to fight through, which is a different and much harder problem.
- Gripping is a proximity-triggered fixed constraint, not simulated finger
  contact — appropriate for a suction-style SCARA end effector.
- The bin coordinates are deliberately kept out of the observation. The policy
  gets only a colour one-hot and must learn the colour→location mapping.
