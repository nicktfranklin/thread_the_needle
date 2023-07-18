import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO

from state_inference.gridworld_env import CnnWrapper, OpenEnv
from state_inference.utils.training_utils import train_model

# Discritized states: a 20x20 grid of states, which we embed by spacing
# evenly in a nXn space
HEIGHT, WIDTH = 20, 20
MAP_HEIGHT = 60

TEST_START_STATE = WIDTH - 1  # Top right corner
TEST_START_STATE = (WIDTH * HEIGHT - 1) // 2 + WIDTH // 2  # center

N_EPOCHS = 100
N_STEPS = 10000
N_EVAL_STEPS = 100

#### for open env
STATE_REWARDS = {0: 10, 399: 10, 19: -1, 380: -1}
END_STATE = {0, 399, 19, 380}
#### end for open env


def train():
    obs_kwargs = dict(
        rbf_kernel_size=51,
        rbf_kernel_scale=0.2,
        location_noise_scale=0.5,  # must be odd
    )

    ## Wrap these in a world model
    args = [HEIGHT, WIDTH, MAP_HEIGHT, STATE_REWARDS, obs_kwargs]
    kwargs = dict(n_states=HEIGHT * WIDTH, end_state=END_STATE)

    task = CnnWrapper(OpenEnv.create_env(*args, **kwargs))

    pi, _ = task.get_optimal_policy()

    train_kwargs = dict(
        optimal_policy=pi,
        n_epochs=N_EPOCHS,
        n_train_steps=N_STEPS,
        n_obs=5,
        n_states=400,
        map_height=MAP_HEIGHT,
        n_eval_steps=N_EVAL_STEPS,
        test_start_state=TEST_START_STATE,
    )

    ppo = PPO("CnnPolicy", task, verbose=0)
    ppo_rewards = train_model(ppo, **train_kwargs)

    a2c = A2C("CnnPolicy", task, verbose=0)
    a2c_rewards = train_model(a2c, **train_kwargs)

    dqn = DQN("CnnPolicy", task, verbose=0)
    dqn_rewards = train_model(dqn, **train_kwargs)

    pd.DataFrame(
        {
            "Rewards": np.concatenate([dqn_rewards, a2c_rewards, ppo_rewards]),
            "Model": ["DQN"] * (N_EPOCHS + 1)
            + ["A2C"] * (N_EPOCHS + 1)
            + ["PPO"] * (N_EPOCHS + 1),
            "Epoch": [ii for ii in range(N_EPOCHS + 1)]
            + [ii for ii in range(N_EPOCHS + 1)]
            + [ii for ii in range(N_EPOCHS + 1)],
        },
    ).set_index("Epoch").to_csv("Simulations.csv")


if __name__ == "__main__":
    train()
