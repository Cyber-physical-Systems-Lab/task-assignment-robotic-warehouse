import gymnasium as gym

class SB3SingleAgentWrapper(gym.Wrapper):
    """
    A wrapper to adapt the custom Warehouse env for Stable-Baselines3.
    Filters reset/step outputs and gives only obs, reward, done, info.
    """

    def reset(self, **kwargs):
        obs_tuple = super().reset(**kwargs)
        obs = obs_tuple[0]  # Only observation
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        return obs, reward, terminated, truncated, info
