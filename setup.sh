#!/usr/bin/env bash
# SCARA pick-and-place: one-shot dependency install.
# Run once:  bash ~/scara_ws/setup.sh
set -e

echo "############ 1/3  apt packages ############"
sudo apt update
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-xacro \
  ros-humble-joint-state-publisher-gui \
  ros-humble-image-transport-plugins \
  python3-colcon-common-extensions \
  python3-pip \
  python3.10-venv \
  python3-transforms3d

echo "############ 2/3  python venv ############"
# --system-site-packages so the venv still sees rclpy / cv_bridge / cv2 from ROS.
if [ ! -d "$HOME/scara_venv" ]; then
  python3 -m venv --system-site-packages "$HOME/scara_venv"
fi
# shellcheck disable=SC1091
source "$HOME/scara_venv/bin/activate"

python -m pip install --upgrade pip wheel
# CPU torch: PPO on a small MLP is faster on CPU than GPU, and the CPU wheel is
# ~1.5 GB smaller -- matters with only 7 GB of RAM in this WSL instance.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pybullet "gymnasium>=0.29" "stable-baselines3>=2.2" tensorboard

echo "############ 3/3  verify ############"
python - <<'PY'
import importlib, sys
ok = True
for m in ("pybullet", "gymnasium", "stable_baselines3", "torch", "cv2", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {m:20s} {e}")
sys.exit(0 if ok else 1)
PY

source /opt/ros/humble/setup.bash
for p in gazebo_ros gazebo_ros2_control controller_manager xacro; do
  if ros2 pkg prefix "$p" >/dev/null 2>&1; then echo "  OK   ros pkg $p"; else echo "  FAIL ros pkg $p"; fi
done
command -v gazebo >/dev/null && echo "  OK   gazebo $(gazebo --version 2>/dev/null | head -1)" || echo "  FAIL gazebo binary"

echo
echo "Setup finished. Activate the RL env in any new shell with:"
echo "    source ~/scara_venv/bin/activate"
