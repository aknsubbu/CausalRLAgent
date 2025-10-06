import gym
from gym import spaces
import nle  # NetHack Learning Environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def make_env():
    return gym.make("NetHackScore-v0")

env = DummyVecEnv([make_env])


class SemanticWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "glyphs": self.env.observation_space["glyphs"],
            "blstats": self.env.observation_space["blstats"],
            "message": spaces.Box(low=0, high=255, shape=(80,), dtype=np.int32),  # cropped message
        })

    def observation(self, obs):
        return {
            "glyphs": obs["glyphs"],
            "blstats": obs["blstats"],
            "message": obs["message"][:80],  # truncate text message
        }

env = SemanticWrapper(gym.make("NetHackScore-v0"))



model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./nethack_tensorboard_rlbasic/"
)


model.learn(total_timesteps=200000)
model.save("ppo_nethack_semantic")

obs = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
