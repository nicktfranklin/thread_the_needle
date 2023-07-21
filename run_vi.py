import numpy as np
import pandas as pd

from state_inference.gridworld_env import CnnWrapper, OpenEnv
from state_inference.model.baseline_compatible import ValueIterationAgent
from state_inference.utils.training_utils import train_model

# Discritized states: a 20x20 grid of states, which we embed by spacing
# evenly in a nXn space
HEIGHT, WIDTH = 20, 20
MAP_HEIGHT = 40

TEST_START_STATE = WIDTH // 2  # Top middle

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
from state_inference.model.vae import DEVICE, Decoder, Encoder, StateVae

### Model + Training Parameters


##### Model Kwargs
def make_vi(task):
    N_EPOCHS = 20  # should be 20
    EMBEDDING_LAYERS = 5

    LR = 3e-4
    beta = 1.0
    tau = 2.0
    gamma = 0.99
    dropout = 0.0
    N_ACTIONS = 4

    EMBEDDING_DIM = len(task.observation_model.states) // 2
    OBSERVATION_DIM = task.observation_model.map_height**2

    # create the model
    encoder_hidden = [OBSERVATION_DIM // 5, OBSERVATION_DIM // 10]
    decoder_hidden = [OBSERVATION_DIM // 10, OBSERVATION_DIM // 5]
    z_dim = EMBEDDING_DIM * EMBEDDING_LAYERS

    encoder = Encoder(
        OBSERVATION_DIM,
        encoder_hidden,
        z_dim,
        dropout=dropout,
    )

    decoder = Decoder(
        z_dim,
        decoder_hidden,
        OBSERVATION_DIM,
        dropout=dropout,
    )

    vae_kwargs = dict(
        z_dim=EMBEDDING_DIM, z_layers=EMBEDDING_LAYERS, beta=beta, tau=tau, gamma=gamma
    )

    vae_model = StateVae(encoder, decoder, **vae_kwargs).to(DEVICE)

    return ValueIterationAgent(task, vae_model, set_action=set(range(N_ACTIONS)))


def train():
    ## Wrap these in a world model
    args = [HEIGHT, WIDTH, MAP_HEIGHT, STATE_REWARDS, OBS_KWARGS]
    kwargs = dict(
        movement_penalty=MOVEMENT_PENALTY, n_states=HEIGHT * WIDTH, end_state=END_STATE
    )

    task = OpenEnv.create_env(*args, **kwargs)

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

    ppo = make_vi(task)
    vi_rewards = train_model(ppo, **train_kwargs)

    pd.DataFrame(
        {
            "Rewards": np.concatenate([vi_rewards]),
            "Model": ["VI"] * (N_EPOCHS + 1),
            "Epoch": [ii for ii in range(N_EPOCHS + 1)],
        },
    ).set_index("Epoch").to_csv("OpenEnvSims.csv")


if __name__ == "__main__":
    train()
