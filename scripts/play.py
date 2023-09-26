import time
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO

from task.gridworld import CnnWrapper, OpenEnv
from utils.training_utils import train_model

# Discritized states: a 20x20 grid of states, which we embed by spacing
# evenly in a nXn space
HEIGHT, WIDTH = 20, 20
MAP_HEIGHT = 40

TEST_START_STATE = WIDTH // 2  # Top middle
# TEST_START_STATE = (WIDTH * HEIGHT - 1) // 2 + WIDTH // 2  # center

N_EPOCHS = 100
N_STEPS = 10000
N_EVAL_STEPS = 100

#### for open env
goal_loc = (WIDTH * HEIGHT - 1) // 2 + WIDTH // 2
STATE_REWARDS = {goal_loc: 10, 0: -1, 399: -1, 19: -1, 380: -1}
END_STATE = {goal_loc, 0, 399, 19, 380}
MOVEMENT_PENALTY = -0.1
#### end for open env

OBS_KWARGS = dict(
    rbf_kernel_size=31,  # must be odd
    rbf_kernel_scale=0.35,
    location_noise_scale=0.25,
    noise_log_mean=-3,
    noise_log_scale=0.05,
    noise_corruption_prob=0.005,
)


def play():
    plt.ion()
    args = [HEIGHT, WIDTH, MAP_HEIGHT, STATE_REWARDS, OBS_KWARGS]
    kwargs = dict(
        movement_penalty=MOVEMENT_PENALTY, n_states=HEIGHT * WIDTH, end_state=END_STATE
    )

    task = OpenEnv.create_env(*args, **kwargs)

    obs = task.reset()
    print(obs)
    plt.imshow(obs[0])
    plt.pause(0.5)

    done = False
    while not done:
        # action = int(input('0: up, 1: down, 2: left, 3: right?'))
        action = choice([0, 1, 2, 3])
        key = {0: "up", 1: "down", 2: "left", 3: "right"}
        obs, rew, done, info, _ = task.step(action)
        print(f"Action: {key[action]}, reward: {rew}")
        plt.imshow(obs)
        plt.pause(0.1)

    # time.sleep(10)


if __name__ == "__main__":
    play()
