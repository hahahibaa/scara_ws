"""Train the colour-sorting SCARA policy with PPO.

    source ~/scara_venv/bin/activate
    cd ~/scara_ws/src/scara_rl
    python -m scara_rl.train_ppo --timesteps 1500000

Watch it learn:
    tensorboard --logdir ~/scara_ws/runs
"""

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from .scara_env import ScaraPickPlaceEnv

WS = Path(__file__).resolve().parents[3]
MODEL_DIR = WS / "models"
LOG_DIR = WS / "runs"


class SuccessRateCallback(BaseCallback):
    """PPO logs reward but not task success, and for a sorting task the
    success rate -- overall and per colour -- is the number that matters."""

    def __init__(self, window=200, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.results = []          # (success, colour_index)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                ci = ["red", "green", "blue"].index(info.get("color", "red"))
                self.results.append((bool(info.get("is_success", False)), ci))
        if len(self.results) > self.window:
            self.results = self.results[-self.window:]

        if self.n_calls % 2048 == 0 and self.results:
            arr = np.array([r[0] for r in self.results], dtype=np.float32)
            cols = np.array([r[1] for r in self.results])
            self.logger.record("task/success_rate", float(arr.mean()))
            for i, name in enumerate(["red", "green", "blue"]):
                m = cols == i
                if m.any():
                    self.logger.record(f"task/success_{name}", float(arr[m].mean()))
        return True


def make_env(rank, seed=0):
    def _init():
        env = ScaraPickPlaceEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv, filename=str(LOG_DIR / "monitor.csv"),
                      info_keywords=("is_success", "color"))

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(LOG_DIR))
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=512,               # 512 * n_envs per update
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,            # the colour->bin mapping needs exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=str(LOG_DIR),
            seed=args.seed,
            verbose=1,
        )

    cbs = [
        SuccessRateCallback(),
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=str(MODEL_DIR / "checkpoints"),
                           name_prefix="scara_ppo"),
    ]

    model.learn(total_timesteps=args.timesteps, callback=cbs,
                progress_bar=False, tb_log_name="ppo")
    out = MODEL_DIR / "scara_ppo_final"
    model.save(str(out))
    print(f"\nsaved -> {out}.zip")
    venv.close()


if __name__ == "__main__":
    # PyBullet spawns its own threads; keep BLAS from oversubscribing the CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
