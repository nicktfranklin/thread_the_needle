import numpy as np
import pandas as pd
import yaml

from state_inference.gridworld_env import OpenEnv
from state_inference.model.baseline_compatible import ValueIterationAgent
from state_inference.utils.training_utils import train_model
from state_inference.model.vae import DEVICE, Decoder, Encoder, StateVae

CONFIG_FILE = 'state_inference/env_config.yml'
TASK_NAME = 'open_env'
TASK_CLASS = OpenEnv


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
    # parse the config file
    with open(CONFIG_FILE) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    env_kwargs = config[TASK_NAME]['env_kwargs']
    obs_kwargs = config['obs_kwargs']
    test_start_state = config[TASK_NAME]['test_start_state']
    training_kwargs = config['training_kwargs']

    # create the task
    task = TASK_CLASS.create_env(**env_kwargs, observation_kwargs=obs_kwargs)

    pi, _ = task.get_optimal_policy()

    train_kwargs = dict(
        optimal_policy=pi,
        n_epochs=training_kwargs['n_epochs'],
        n_train_steps=training_kwargs['n_steps'],
        n_obs=5,
        n_states=env_kwargs['n_states'],
        map_height=env_kwargs['map_height'],
        n_eval_steps=training_kwargs['n_eval_steps'],
        test_start_state=test_start_state,
    )

    ppo = make_vi(task)
    vi_rewards = train_model(ppo, **train_kwargs)

    pd.DataFrame(
        {
            "Rewards": np.concatenate([vi_rewards]),
            "Model": ["VI"] * (config['training_kwargs']['n_epochs'] + 1),
            "Epoch": [ii for ii in range(config['training_kwargs']['n_epochs'] + 1)],
        },
    ).set_index("Epoch").to_csv("OpenEnvSims.csv")


if __name__ == "__main__":
    train()
