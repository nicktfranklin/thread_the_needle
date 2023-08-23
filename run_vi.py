import numpy as np
import pandas as pd
import yaml

from state_inference.gridworld_env import ThreadTheNeedleEnv
from state_inference.model.agents import ViAgentWithExploration
from state_inference.model.vae import DEVICE, MlpDecoder, MlpEncoder, StateVae
from state_inference.utils.training_utils import parse_task_config, train_model

CONFIG_FILE = "state_inference/env_config.yml"
TASK_NAME = "thread_the_needle"
TASK_CLASS = ThreadTheNeedleEnv
OUTPUT_FILE_NAME = "ThreadTheNeedleSims.csv"


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

    encoder = MlpEncoder(
        OBSERVATION_DIM,
        encoder_hidden,
        z_dim,
        dropout=dropout,
    )

    decoder = MlpDecoder(
        z_dim,
        decoder_hidden,
        OBSERVATION_DIM,
        dropout=dropout,
    )

    vae_kwargs = dict(
        z_dim=EMBEDDING_DIM, z_layers=EMBEDDING_LAYERS, beta=beta, tau=tau, gamma=gamma
    )

    vae_model = StateVae(encoder, decoder, **vae_kwargs).to(DEVICE)

    return ViAgentWithExploration(task, vae_model, set_action=set(range(N_ACTIONS)))


def train():
    env_kwargs, training_kwargs = parse_task_config(TASK_NAME, CONFIG_FILE)

    # create the task
    task = TASK_CLASS.create_env(**env_kwargs)

    pi, _ = task.get_optimal_policy()
    training_kwargs["optimal_policy"] = pi

    results = []

    def append_res(results, rewards, model_name):
        results.append(
            {
                "Rewards": rewards,
                "Model": [model_name] * (training_kwargs["n_epochs"] + 1),
                "Epoch": [ii for ii in range(training_kwargs["n_epochs"] + 1)],
            }
        )

    agent = make_vi(task)
    vi_rewards = train_model(agent, **training_kwargs)

    append_res(results, vi_rewards, "Agent")

    results = pd.concat([pd.DataFrame(res) for res in results])
    results.set_index("Epoch").to_csv(OUTPUT_FILE_NAME)


if __name__ == "__main__":
    train()
