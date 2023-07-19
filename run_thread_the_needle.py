import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DQN, PPO

from state_inference.gridworld_env import CnnWrapper, ThreadTheNeedleEnv
from state_inference.utils.training_utils import train_model

# Discritized states: a 20x20 grid of states, which we embed by spacing
# evenly in a nXn space
HEIGHT, WIDTH = 20, 20
MAP_HEIGHT = 40

TEST_START_STATE = WIDTH - 1  # Top right corner
# TEST_START_STATE = (WIDTH * HEIGHT - 1) // 2 + WIDTH // 2  # center

N_EPOCHS = 100
N_STEPS = 10000
N_EVAL_STEPS = 100

#### for open env
STATE_REWARDS = {0: 10}
END_STATE = {0}
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


def train():
    ## Wrap these in a world model
    args = [HEIGHT, WIDTH, MAP_HEIGHT, STATE_REWARDS, OBS_KWARGS]
    kwargs = dict(
        movement_penalty=MOVEMENT_PENALTY, n_states=HEIGHT * WIDTH, end_state=END_STATE
    )

    task = CnnWrapper(ThreadTheNeedleEnv.create_env(*args, **kwargs))

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
    ).set_index("Epoch").to_csv("ThreadTheNeedleSims.csv")


if __name__ == "__main__":
    train()
