import os
import sys
# Add the root directory to sys.path so tarware can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tarware.utils.sb3_wrapper import SB3SingleAgentWrapper


from tarware.warehouse import Warehouse
from tarware.utils.wrappers import AGVWrapper

# Parameters 
TOTAL_TIMESTEPS = 10_000
SAVE_PATH = "ppo_agv_model"

def make_env():
    # Create the Warehouse env
    env = Warehouse(
        shelf_columns=5,
        column_height=4,
        shelf_rows=4,
        num_agvs=1,               # Only training 1 AGV
        num_pickers=0,            # No pickers for now
        request_queue_size=3,
        max_inactivity_steps=100,
        max_steps=500,
        reward_type="INDIVIDUAL",  # Better for PPO in single-agent
        normalised_coordinates=False,
        observation_type="global"
    )

    # Filter to AGV only
    env = AGVWrapper(env)

    env = SB3SingleAgentWrapper(env)

    # Wrap with Monitor for logging
    env = Monitor(env)

    # Wrap for vectorization (required by SB3)
    env = DummyVecEnv([lambda: env])

    return env

def main():
    env = make_env()

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ppo_agv_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save the model
    model.save(SAVE_PATH)
    print(f"\n Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
