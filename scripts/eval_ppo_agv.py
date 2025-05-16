import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add tarware to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tarware.utils.wrappers import AGVWrapper

# Custom Gym Wrapper to adapt reset return value
class ResetOnlyObsWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]  # discard info
        return result

MODEL_PATH = "ppo_agv_model"
model = PPO.load(MODEL_PATH)

# Env builder
def make_env():
    def _init():
        env = gym.make("tarware-extralarge-14agvs-7pickers-partialobs-v1")
        env = AGVWrapper(env)
        env = ResetOnlyObsWrapper(env)
        return env
    return _init

# DummyVecEnv expects a list of functions
env = DummyVecEnv([make_env()])

# Safe to call reset now
obs = env.reset()

# Run simulation
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = np.logical_or(terminated, truncated)

    env.envs[0].render(mode="human")
    time.sleep(0.1)

    if done[0]:
        print("Episode done. Resetting.")
        obs = env.reset()
