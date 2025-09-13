#!/usr/bin/env python3
"""
train_nethack_fixed_smallcnn.py

- gymnasium-based NetHack pipeline
- uint8 channel-first image obs (1,21,79)
- custom small CNN features extractor for small images (avoids NatureCNN size assumptions)
- human-readable map utilities + map dump to ./maps/
- training (PPO/A2C/DQN), EvalCallback, Checkpointing, Benchmarking, plotting

Run: python train_nethack_fixed_smallcnn.py --timesteps 50000
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Try import nle
try:
    import nle  # type: ignore
    from nle import nethack  # type: ignore
except Exception:
    nle = None
    nethack = None

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("nethack_smallcnn")

# BLSTATS index map (adjust for NLE version if required)
BLSTATS_IDX = {
    "x": 0,
    "y": 1,
    "score": 9,
    "hp": 10,
    "hp_max": 11,
    "depth": 12,
    "experience": 18,
}


def safe_get_blstat(blstats: Optional[np.ndarray], name: str, default: int = 0) -> int:
    idx = BLSTATS_IDX.get(name)
    if idx is None or blstats is None:
        return default
    try:
        if len(blstats) <= idx:
            return default
        return int(blstats[idx])
    except Exception:
        return default


DEFAULT_REWARD_CONFIG = {
    "score_scale": 0.01,
    "heal_reward": 0.1,
    "damage_penalty": -0.1,
    "death_penalty": -10.0,
    "explore_reward": 0.05,
    "depth_reward": 5.0,
    "exp_scale": 0.001,
    "time_penalty": -0.001,
    "survival_bonus": 0.002,
}

# ---------------------------
# Custom small CNN extractor
# ---------------------------
class SmallCnnExtractor(BaseFeaturesExtractor):
    """
    Small CNN features extractor for small channel-first images such as (1,21,79).

    Produces a fixed-size vector by:
      - conv3x3 (padding=1) blocks to preserve spatial dims
      - adaptive avg pool to (1,1)
    Normalizes uint8->float32 [0,1] internally.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        # observation_space shape expected (C, H, W)
        super().__init__(observation_space, features_dim)
        shape = observation_space.shape
        if len(shape) != 3:
            raise ValueError("SmallCnnExtractor expects a channel-first image shape: (C,H,W)")
        in_channels = shape[0]

        # small conv stack
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Reduce to single spatial cell
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # final linear to features_dim
        self.linear = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

        # set _features_dim to features_dim so SB3 knows the output size
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations expected as torch tensor with shape (batch, C, H, W)
        # if uint8, convert and normalize
        x = observations.float()
        # If user kept uint8 range 0..255, normalize
        if x.max() > 1.0:
            x = x / 255.0
        x = self.cnn(x)
        x = self.linear(x)
        return x


# ---------------------------
# Human-readable utilities (same as earlier)
# ---------------------------
def _try_nle_glyph_to_ascii(glyph: int) -> Optional[str]:
    try:
        if nethack is None:
            return None
        for name in ("glyph2ascii", "glyph_to_ascii", "glyph2char", "glyph_to_char"):
            if hasattr(nethack, name):
                fn = getattr(nethack, name)
                try:
                    res = fn(glyph)
                    if isinstance(res, bytes):
                        res = res.decode("utf-8", errors="replace")
                    if isinstance(res, str) and len(res) > 0:
                        return res[0]
                except Exception:
                    continue
    except Exception:
        pass
    return None


def glyphs_to_printable(glyphs: np.ndarray) -> np.ndarray:
    glyphs = np.asarray(glyphs)
    if glyphs.ndim != 2:
        if glyphs.ndim == 3 and glyphs.shape[0] == 1:
            glyphs = glyphs[0]
        elif glyphs.ndim == 3 and glyphs.shape[2] == 1:
            glyphs = glyphs[:, :, 0]
        else:
            try:
                glyphs = glyphs.reshape((21, 79))
            except Exception:
                glyphs = np.zeros((21, 79), dtype=np.int32)

    H, W = glyphs.shape
    out = np.full((H, W), " ", dtype=object)
    if np.issubdtype(glyphs.dtype, np.integer) and glyphs.max() <= 127 and glyphs.min() >= 0:
        for i in range(H):
            for j in range(W):
                v = int(glyphs[i, j])
                out[i, j] = chr(v) if 32 <= v <= 126 else " "
        return out

    unique_vals = np.unique(glyphs)
    cache: Dict[int, str] = {}
    for val in unique_vals:
        if val == 0:
            cache[int(val)] = " "
            continue
        ascii_ch = _try_nle_glyph_to_ascii(int(val))
        if ascii_ch is not None:
            cache[int(val)] = ascii_ch
        else:
            if 32 <= val <= 126:
                cache[int(val)] = chr(val)
            elif val < 256:
                cache[int(val)] = "."
            else:
                cache[int(val)] = "?"
    for i in range(H):
        for j in range(W):
            out[i, j] = cache[int(glyphs[i, j])]
    return out


def vector_to_grid(vec: np.ndarray, height: int = 21, width: int = 79) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 1 and arr.size == height * width:
        return arr.reshape((height, width))
    try:
        return arr.reshape((height, width))
    except Exception:
        return np.zeros((height, width), dtype=np.int32)


def blstats_to_readable(blstats: Optional[np.ndarray]) -> Dict[str, Any]:
    if blstats is None:
        return {}
    arr = np.array(blstats)
    return {
        "x": safe_get_blstat(arr, "x"),
        "y": safe_get_blstat(arr, "y"),
        "score": safe_get_blstat(arr, "score"),
        "hp": safe_get_blstat(arr, "hp"),
        "hp_max": safe_get_blstat(arr, "hp_max"),
        "depth": safe_get_blstat(arr, "depth"),
        "experience": safe_get_blstat(arr, "experience"),
    }


def obs_to_human_readable(obs: Any, height: int = 21, width: int = 79, include_blstats: bool = True) -> Dict[str, Any]:
    raw = obs
    glyphs = None
    blstats = None

    if isinstance(raw, dict):
        if "tty_chars" in raw and isinstance(raw["tty_chars"], np.ndarray):
            try:
                arr = np.array(raw["tty_chars"])
                H, W = arr.shape[:2]
                grid = np.empty((H, W), dtype=object)
                for i in range(H):
                    for j in range(W):
                        v = int(arr[i, j])
                        grid[i, j] = chr(v) if 32 <= v <= 126 else " "
                text_map = "\n".join("".join(row.tolist()) for row in grid)
                return {"text_map": text_map, "grid_chars": grid, "blstats": blstats_to_readable(raw.get("blstats")) if include_blstats else None}
            except Exception:
                pass

        if "tty_glyphs" in raw and isinstance(raw["tty_glyphs"], np.ndarray):
            glyphs = np.array(raw["tty_glyphs"])
        elif "glyphs" in raw and isinstance(raw["glyphs"], np.ndarray):
            glyphs = np.array(raw["glyphs"])
        blstats = raw.get("blstats", None)
    elif isinstance(raw, np.ndarray):
        glyphs = raw
    elif isinstance(raw, (tuple, list)) and len(raw) >= 1:
        cand = raw[0]
        if isinstance(cand, dict):
            return obs_to_human_readable(cand, height=height, width=width, include_blstats=include_blstats)
        if isinstance(cand, np.ndarray):
            glyphs = cand

    if glyphs is None:
        glyphs = np.zeros((height, width), dtype=np.int32)

    grid = vector_to_grid(glyphs, height=height, width=width)
    grid_chars = glyphs_to_printable(grid)
    text_map = "\n".join("".join(row.tolist()) for row in grid_chars)

    return {"text_map": text_map, "grid_chars": grid_chars, "blstats": blstats_to_readable(blstats) if include_blstats else None}


# ---------------------------
# NetHackWrapper (image uint8 obs)
# ---------------------------
class NetHackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, render_mode: Optional[str] = None, reward_config: Optional[Dict[str, float]] = None, use_image_obs: bool = True, normalize_images: bool = False, max_episode_steps: int = 5000):
        super().__init__(env)
        self._render_mode = render_mode
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG.copy()
        self.use_image_obs = bool(use_image_obs)
        self.normalize_images = bool(normalize_images)
        self.max_episode_steps = int(max_episode_steps)
        self.visited_positions = set()
        self.prev_blstats: Optional[np.ndarray] = None
        self.step_count = 0

        glyph_shape = (21, 79)
        if self.use_image_obs:
            if self.normalize_images:
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, glyph_shape[0], glyph_shape[1]), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=(1, glyph_shape[0], glyph_shape[1]), dtype=np.uint8)
        else:
            if self.normalize_images:
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(glyph_shape[0] * glyph_shape[1],), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=(glyph_shape[0] * glyph_shape[1],), dtype=np.uint8)

        self.action_space = env.action_space

    def _extract_glyphs(self, raw_obs: Any) -> np.ndarray:
        if raw_obs is None:
            return np.zeros((21, 79), dtype=np.int32)
        if isinstance(raw_obs, dict):
            for key in ("glyphs", "tty_glyphs", "tty_chars"):
                val = raw_obs.get(key)
                if isinstance(val, np.ndarray):
                    arr = np.array(val, copy=False)
                    if arr.ndim > 2:
                        arr = arr[0]
                    return arr.astype(np.int32)
        if isinstance(raw_obs, np.ndarray) and raw_obs.ndim >= 2:
            return raw_obs.astype(np.int32)
        return np.zeros((21, 79), dtype=np.int32)

    def _format_observation(self, raw_obs: Any) -> np.ndarray:
        glyphs = self._extract_glyphs(raw_obs)
        glyphs8 = (glyphs % 256).astype(np.uint8)
        if self.use_image_obs:
            img = np.expand_dims(glyphs8, axis=0)  # (1,H,W)
            if self.normalize_images:
                return (img.astype(np.float32) / 255.0).astype(np.float32)
            else:
                return img
        else:
            flat = glyphs8.flatten()
            if self.normalize_images:
                return (flat.astype(np.float32) / 255.0).astype(np.float32)
            else:
                return flat

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        self.step_count = 0
        self.visited_positions.clear()
        self.prev_blstats = None
        if isinstance(obs, dict) and "blstats" in obs:
            try:
                self.prev_blstats = np.array(obs["blstats"], copy=True)
                x = safe_get_blstat(self.prev_blstats, "x")
                y = safe_get_blstat(self.prev_blstats, "y")
                self.visited_positions.add((x, y))
            except Exception:
                self.prev_blstats = None
        formatted = self._format_observation(obs)
        return formatted, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, base_reward, terminated, truncated, info = result
        else:
            obs, base_reward, done, info = result
            terminated = bool(done)
            truncated = False
        self.step_count += 1
        shaped = self._calculate_shaped_reward(obs, float(base_reward or 0.0), terminated, truncated, info)
        if self.step_count >= self.max_episode_steps:
            truncated = True
        return self._format_observation(obs), shaped, terminated, truncated, info

    def _calculate_shaped_reward(self, obs, base_reward, terminated, truncated, info) -> float:
        cfg = self.reward_config
        reward = float(base_reward or 0.0)
        raw_bl = obs.get("blstats") if isinstance(obs, dict) else None
        if raw_bl is None or self.prev_blstats is None:
            if raw_bl is not None:
                try:
                    self.prev_blstats = np.array(raw_bl, copy=True)
                except Exception:
                    self.prev_blstats = None
            reward += cfg["time_penalty"]
            if not terminated and not truncated:
                reward += cfg["survival_bonus"]
            return reward

        cur = np.array(raw_bl, copy=True)
        prev = np.array(self.prev_blstats, copy=True)
        score_diff = safe_get_blstat(cur, "score") - safe_get_blstat(prev, "score")
        if score_diff > 0:
            reward += score_diff * cfg["score_scale"]
        hp_prev = safe_get_blstat(prev, "hp")
        hp_cur = safe_get_blstat(cur, "hp")
        if hp_cur > hp_prev:
            reward += cfg["heal_reward"]
        elif hp_cur < hp_prev:
            reward += cfg["damage_penalty"]
        if terminated and hp_cur <= 0:
            reward += cfg["death_penalty"]
        x = safe_get_blstat(cur, "x")
        y = safe_get_blstat(cur, "y")
        if (x, y) not in self.visited_positions:
            self.visited_positions.add((x, y))
            reward += cfg["explore_reward"]
        depth_diff = safe_get_blstat(cur, "depth") - safe_get_blstat(prev, "depth")
        if depth_diff > 0:
            reward += cfg["depth_reward"] * depth_diff
        exp_diff = safe_get_blstat(cur, "experience") - safe_get_blstat(prev, "experience")
        if exp_diff > 0:
            reward += exp_diff * cfg["exp_scale"]
        reward += cfg["time_penalty"]
        if not terminated and not truncated:
            reward += cfg["survival_bonus"]
        try:
            self.prev_blstats = cur.copy()
        except Exception:
            self.prev_blstats = None
        return float(reward)

    def render(self, mode: str = "human"):
        if mode != "human":
            return super().render(mode=mode)
        last_obs = getattr(self.env.unwrapped, "last_observation", None)
        if last_obs is None:
            try:
                last_obs, _ = self.env.reset()
            except Exception:
                last_obs = None
        if last_obs and isinstance(last_obs, dict):
            chars = last_obs.get("tty_chars") or last_obs.get("tty_glyphs") or None
            if isinstance(chars, np.ndarray):
                for row in chars:
                    print("".join([chr(c) if 32 <= c <= 126 else " " for c in row]))
            print(f"Score: {last_obs.get('score', 'n/a')} Turn: {last_obs.get('turn', 'n/a')}")
        return super().render(mode=mode)

    def human_readable(self, obs: Optional[Any] = None, include_blstats: bool = True) -> Dict[str, Any]:
        if obs is None:
            obs = getattr(self.env.unwrapped, "last_observation", None)
            if obs is None:
                try:
                    obs, _ = self.env.reset()
                except Exception:
                    obs = None
        return obs_to_human_readable(obs, include_blstats=include_blstats)

    def human_readable_dump(self, obs: Optional[Any] = None, directory: str = "maps", filename: Optional[str] = None, include_blstats: bool = True) -> str:
        hr = self.human_readable(obs=obs, include_blstats=include_blstats)
        text_map = hr.get("text_map", "")
        bl = hr.get("blstats", {})
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            score = bl.get("score", "n_a")
            depth = bl.get("depth", "n_a")
            filename = f"map_{stamp}_score{score}_depth{depth}.txt"
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Human-readable NetHack map dump\n")
            f.write(f"# timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# blstats: {json.dumps(bl)}\n\n")
            f.write(text_map)
            f.write("\n")
        return path


# ---------------------------
# Callbacks & env factory & training
# ---------------------------
class BenchmarkCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            for info in infos:
                ep = info.get("episode")
                if ep:
                    self.episode_rewards.append(float(ep.get("r", 0.0)))
                    self.episode_lengths.append(int(ep.get("l", 0)))
        return True

    def get_stats(self) -> Dict[str, float]:
        total_time = time.time() - self.start_time if self.start_time else 0.0
        return {
            "total_training_time": total_time,
            "total_episodes": len(self.episode_rewards),
            "avg_episode_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "best_episode_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
        }


def create_nethack_env(render_mode: Optional[str] = None, reward_config: Optional[Dict[str, float]] = None, use_image_obs: bool = True, normalize_images: bool = False, max_episode_steps: int = 5000, seed: Optional[int] = None) -> gym.Env:
    env = gym.make("NetHackScore-v0")
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass
    return NetHackWrapper(env, render_mode=render_mode, reward_config=reward_config, use_image_obs=use_image_obs, normalize_images=normalize_images, max_episode_steps=max_episode_steps)


def make_vec_nethack_env(n_envs: int = 1, reward_config: Optional[Dict[str, float]] = None, use_image_obs: bool = True, normalize_images: bool = False, seed: Optional[int] = None) -> gym.Env:
    return make_vec_env(lambda: Monitor(create_nethack_env(render_mode=None, reward_config=reward_config, use_image_obs=use_image_obs, normalize_images=normalize_images, seed=seed)), n_envs=n_envs)


def train_agent(algorithm: str = "PPO", total_timesteps: int = 100_000, n_envs: int = 1, use_image_obs: bool = True, normalize_images: bool = False, eval_freq: int = 5_000, save_dir: str = "./models", model_name: str = "nethack_agent", seed: Optional[int] = None, reward_config: Optional[Dict[str, float]] = None):
    os.makedirs(save_dir, exist_ok=True)
    set_random_seed(seed or 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("Training: alg=%s device=%s timesteps=%d", algorithm, device, total_timesteps)

    train_env = make_vec_nethack_env(n_envs=n_envs, reward_config=reward_config, use_image_obs=use_image_obs, normalize_images=normalize_images, seed=seed)
    eval_env = Monitor(create_nethack_env(render_mode=None, reward_config=reward_config, use_image_obs=use_image_obs, normalize_images=normalize_images, seed=(seed + 999 if seed else None)))

    if use_image_obs:
        policy = "CnnPolicy"
        # tell SB3 to use our extractor
        policy_kwargs = {
            "features_extractor_class": SmallCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
    else:
        policy = "MlpPolicy"
        policy_kwargs = {}

    common_kwargs = {"verbose": 1, "tensorboard_log": os.path.join(save_dir, "tensorboard"), "device": device, "policy_kwargs": policy_kwargs}

    if algorithm.upper() == "PPO":
        model = PPO(policy, train_env, **common_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    elif algorithm.upper() == "A2C":
        model = A2C(policy, train_env, **common_kwargs)
    elif algorithm.upper() == "DQN":
        model = DQN(policy, train_env, **common_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(save_dir, "best"), log_path=os.path.join(save_dir, "eval"), eval_freq=eval_freq, deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=max(1, total_timesteps // 10), save_path=os.path.join(save_dir, "checkpoints"), name_prefix=model_name)
    benchmark_cb = BenchmarkCallback(verbose=0)

    LOG.info("Starting learning...")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback, benchmark_cb])

    final_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_path)
    LOG.info("Saved final model to %s", final_path)

    stats = benchmark_cb.get_stats()
    return model, stats, benchmark_cb


def benchmark_agent(model, num_episodes: int = 5, render: bool = True, dump_maps: bool = True, maps_dir: str = "maps", use_image_obs: bool = True, normalize_images: bool = False):
    env = create_nethack_env(render_mode="human" if render else None, use_image_obs=use_image_obs, normalize_images=normalize_images)
    env = Monitor(env)

    episode_rewards = []
    episode_lengths = []
    episode_times = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        start = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1
            if render:
                env.render()
                time.sleep(0.03)

        dt = time.time() - start
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        episode_times.append(dt)
        LOG.info("Episode %d: reward=%.3f len=%d time=%.2fs", ep + 1, ep_reward, ep_len, dt)

        if dump_maps:
            try:
                last_obs = getattr(env.unwrapped, "last_observation", None)
                path = env.unwrapped.human_readable_dump(obs=last_obs, directory=maps_dir)
                LOG.info("Dumped human-readable map for episode %d -> %s", ep + 1, path)
            except Exception as e:
                LOG.warning("Failed to dump map for episode %d: %s", ep + 1, e)

    stats = {
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "avg_time_per_episode": float(np.mean(episode_times)) if episode_times else 0.0,
        "best_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "worst_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
    }
    return stats, episode_rewards, episode_lengths


def plot_training_progress(callback: BenchmarkCallback, out_path: str = "training_progress.png"):
    if not callback.episode_rewards:
        LOG.warning("No episode reward data to plot.")
        return
    plt.figure(figsize=(10, 4))
    plt.plot(callback.episode_rewards)
    plt.title("Episode rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


# ---- Test util and CLI ----
def test_env_setup():
    LOG.info("Testing environment...")
    try:
        if nle is None:
            LOG.warning("NLE (nethack) not imported; ensure 'nle' is installed.")
        env = create_nethack_env(render_mode=None)
        obs, info = env.reset()
        LOG.info("Reset ok. obs type=%s", type(obs))
        LOG.info("Obs space=%s action space=%s", env.observation_space, env.action_space)
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            LOG.info("Step %d: reward=%.3f done=%s obs_shape=%s", i + 1, float(reward), bool(terminated or truncated), np.shape(obs))
            if terminated or truncated:
                env.reset()
        env.close()
        return True
    except Exception as e:
        LOG.exception("Environment test failed: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "A2C", "DQN"])
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--use_image_obs", action="store_true", default=True)
    parser.add_argument("--normalize_images", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--model_name", default="nethack_agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark_episodes", type=int, default=2)
    args = parser.parse_args()

    if not test_env_setup():
        LOG.error("Environment setup failed. Please check NLE installation.")
        return

    model, stats, callback = train_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        use_image_obs=args.use_image_obs,
        normalize_images=args.normalize_images,
        eval_freq=args.eval_freq,
        save_dir="./models",
        model_name=args.model_name,
        seed=args.seed,
        reward_config=DEFAULT_REWARD_CONFIG,
    )

    LOG.info("Training stats: %s", stats)
    plot_training_progress(callback)

    LOG.info("Benchmarking model (and dumping maps)...")
    bench_stats, rewards, lengths = benchmark_agent(model, num_episodes=args.benchmark_episodes, render=False, dump_maps=True, maps_dir="maps", use_image_obs=args.use_image_obs, normalize_images=args.normalize_images)
    LOG.info("Benchmark: %s", bench_stats)

    # demo map dump
    env = create_nethack_env(render_mode=None, use_image_obs=args.use_image_obs, normalize_images=args.normalize_images)
    obs, info = env.reset()
    try:
        path = env.human_readable_dump(obs=env.unwrapped.last_observation, directory="maps")
        LOG.info("Demo map dumped -> %s", path)
    except Exception as e:
        LOG.warning("Failed to dump demo map: %s", e)

    LOG.info("Done. Map files placed under ./maps/")


if __name__ == "__main__":
    main()
