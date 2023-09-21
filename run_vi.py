import os
from datetime import date

import numpy as np
import pandas as pd
import torch
import yaml

from state_inference.gridworld_env import CnnWrapper, ThreadTheNeedleEnv
from state_inference.model.agents import ViAgentWithExploration
from state_inference.model.vae import MlpDecoder, MlpEncoder, StateVae
from state_inference.utils.config_utils import (
    load_config,
    parse_model_config,
    parse_task_config,
)
from state_inference.utils.pytorch_utils import (
    DEVICE,
    convert_8bit_to_float,
    make_tensor,
)
from state_inference.utils.training_utils import get_policy_prob, vae_get_pmf

######## File Names and Configs ########
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CONFIG_PATH = "state_inference/configs"
TASK_CONFIG_FILE = "env_config.yml"
VAE_CONFIG_FILE = "vae_config.yml"
AGENT_CONFIG_FILE = "agent_config.yml"

TASK_NAME = "thread_the_needle"
MODEL_NAME = "cnn_vae"

TASK_CLASS = ThreadTheNeedleEnv
AgentClass = ViAgentWithExploration

SIMULATIONS_PATH = "simulations"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def get_value_function(model, task):
    obs = convert_8bit_to_float(
        torch.stack(
            [
                make_tensor(task.observation_model(s))
                for s in range(task.transition_model.n_states)
                for _ in range(1)
            ]
        )
    )[:, None, ...].to(DEVICE)
    z = model.state_inference_model.get_state(obs)

    hash_vector = np.array(
        [
            model.state_inference_model.z_dim**ii
            for ii in range(model.state_inference_model.z_layers)
        ]
    )

    z = z.dot(hash_vector)

    value_function = np.array(
        [model.value_function.get(z0, np.nan) for z0 in z]
    ).reshape(20, 20)
    return value_function


def simulate_agent(task, agent_config, vae_config, env_kwargs, get_pmf_fn: callable):
    agent = AgentClass.make_from_configs(task, agent_config, vae_config, env_kwargs)
    agent.learn(total_timesteps=agent_config["n_train_steps"], progress_bar=True)

    pmf = get_policy_prob(
        agent,
        get_pmf_fn,
        n_states=env_kwargs["n_states"],
        map_height=env_kwargs["map_height"],
        cnn=True,
    )

    v = get_value_function(agent, task)

    # remove the model from memory
    for param in agent.state_inference_model.parameters():
        param.detach()
    agent.state_inference_model = None
    torch.cuda.empty_cache()

    return pmf, v


def train_batches():
    ## Load Configs
    task_config_file = os.path.join(CONFIG_PATH, TASK_CONFIG_FILE)
    vae_config_file = os.path.join(CONFIG_PATH, VAE_CONFIG_FILE)
    agent_config_file = os.path.join(CONFIG_PATH, AGENT_CONFIG_FILE)

    env_kwargs = parse_task_config(TASK_NAME, task_config_file)
    vae_config = parse_model_config(MODEL_NAME, vae_config_file)
    agent_config = load_config(agent_config_file)

    # create the task and get the optimal policy
    task = CnnWrapper(TASK_CLASS.create_env(**env_kwargs))
    pi, _ = task.get_optimal_policy()

    # label each room with a mask for scoring function
    room_1_mask = (np.arange(400) < 200) * (np.arange(400) % 20 < 10)
    room_2_mask = (np.arange(400) >= 200) * (np.arange(400) % 20 < 10)
    room_3_mask = np.arange(400) % 20 >= 10

    scores = []
    value_functions = []

    n_batches = agent_config["n_batches"]

    for idx in range(n_batches):
        print(f"Training Model {idx+1} of {n_batches}")
        pmf, v = simulate_agent(task, agent_config, vae_config, env_kwargs, vae_get_pmf)

        score_room_1 = np.sum(pi[room_1_mask] * pmf[room_1_mask], axis=1).mean()
        score_room_2 = np.sum(pi[room_2_mask] * pmf[room_2_mask], axis=1).mean()
        score_room_3 = np.sum(pi[room_3_mask] * pmf[room_3_mask], axis=1).mean()

        scores.append(
            pd.DataFrame(
                {
                    "Iteration": [idx] * 4,
                    "Score": [
                        np.sum(pi * pmf, axis=1).mean(),
                        score_room_1,
                        score_room_2,
                        score_room_3,
                    ],
                    "Condition": ["Overall", "Room 1", "Room 2", "Room 3"],
                }
            )
        )

        value_functions.append(
            pd.DataFrame(
                {
                    "Iteration": [idx] * task.n_states,
                    "State-Values": v.reshape(-1),
                    "States": np.arange(task.n_states),
                }
            )
        )

    scores = pd.concat(scores)
    value_functions = pd.concat(value_functions)

    # save scores
    scores.to_csv(
        os.path.join(
            SIMULATIONS_PATH, f"scores_{TASK_NAME}_{MODEL_NAME}_{date.today()}.csv"
        )
    )

    # save value function
    value_functions.to_csv(
        os.path.join(
            SIMULATIONS_PATH, f"value_func_{TASK_NAME}_{MODEL_NAME}_{date.today()}.csv"
        )
    )

    # save parameters
    simulations_parameters = {
        "Agent": agent_config,
        "Task": env_kwargs,
        "VAE": vae_config,
    }
    param_file = os.path.join(
        SIMULATIONS_PATH, f"params_{TASK_NAME}_{MODEL_NAME}_{date.today()}.yml"
    )
    with open(param_file, "w") as f:
        yaml.dump(simulations_parameters, f)


if __name__ == "__main__":
    train_batches()
