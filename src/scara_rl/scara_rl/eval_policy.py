"""Evaluate / watch the trained SCARA policy.

    source ~/scara_venv/bin/activate
    cd ~/scara_ws/src/scara_rl

    # numbers only, per colour, headless
    python -m scara_rl.eval_policy --episodes 90

    # watch it in the PyBullet GUI
    python -m scara_rl.eval_policy --render --episodes 10
"""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .scara_env import ScaraPickPlaceEnv
from . import scara_kinematics as K

WS = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = WS / "models" / "scara_ppo_final.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--episodes", type=int, default=90)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = ScaraPickPlaceEnv(render_mode="human" if args.render else None)

    per_color = {c: [] for c in K.COLORS}
    steps_to_success = []

    for ep in range(args.episodes):
        # cycle the colours so each gets an equal share of the episodes
        env.fixed_color = ep % 3
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        n = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            n += 1
            if args.render:
                time.sleep(1.0 / 30)
        ok = bool(info["is_success"])
        per_color[info["color"]].append(ok)
        if ok:
            steps_to_success.append(n)
        if args.render:
            print(f"  ep {ep:3d}  {info['color']:5s}  "
                  f"{'SUCCESS' if ok else 'fail':8s}  {n} steps")

    print("\n---- results ----")
    allr = []
    for c in K.COLORS:
        v = per_color[c]
        allr += v
        if v:
            print(f"  {c:5s}: {100 * np.mean(v):5.1f} %   (n={len(v)})")
    print(f"  {'total':5s}: {100 * np.mean(allr):5.1f} %   (n={len(allr)})")
    if steps_to_success:
        print(f"  mean steps to place: {np.mean(steps_to_success):.0f} "
              f"({np.mean(steps_to_success) / 30:.1f} s)")
    env.close()


if __name__ == "__main__":
    main()
