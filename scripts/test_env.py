import sys
import os
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tarware.warehouse import Warehouse
from tarware.definitions import RewardType  # Correct reward type usage!

def run_env():
    env = Warehouse(
        shelf_columns=3,
        column_height=3,
        shelf_rows=2,
        num_agvs=2,
        num_pickers=2,
        request_queue_size=4,
        max_inactivity_steps=10,
        max_steps=50,
        reward_type=RewardType.GLOBAL,  # Fix this (must use Enum, not string)
        observation_type="global"       # Optional: ensures you're using correct observation
    )

    obs = env.reset()
    print("Initial Observation:", obs)

    done = False
    step = 0

    while not done and step < 30:
        action = env.action_space.sample()  # Random actions

        obs, reward, terminated, truncated, info = env.step(action)

        # Add this to visually render each step
        env.render(mode="human")
        time.sleep(0.5)

        print(f"\n--- Step {step} ---")
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        step += 1

    env.close()
    print("\n Environment run complete.")

if __name__ == "__main__":
    run_env()
