import argparse
import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor, load_results

from model.vae import MlpDecoder, MlpEncoder, StateVae
from model.agents.value_iteration import ValueIterationAgent
from task.gridworld import CnnWrapper, ThreadTheNeedleEnv
from utils.config_utils import parse_configs
from utils.pytorch_utils import DEVICE

print(f"python {sys.version}")
print(f"torch {torch.__version__}")
print(f"device = {DEVICE}")


### Configuration files
parser = argparse.ArgumentParser()

parser.add_argument("--vae_config", default="configs/vae_config.yaml")
parser.add_argument("--task_config", default="config/env_config.yaml")
parser.add_argument("--agent_config", default="config/agent_config.yaml")
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")

# Create log dir
LOG_DIR = "tmp/"
os.makedirs(LOG_DIR, exist_ok=True)

TASK_CLASS = ThreadTheNeedleEnv
AgentClass = ValueIterationAgent

## Load Configs
config = parse_configs(parser.parse_args())

# create the task and get the optimal policy
task = CnnWrapper(TASK_CLASS.create_env(**config["env_kwargs"]))
pi, _ = task.get_optimal_policy()

SAVE_FILE_NAME = f"simulations/thread_the_needle_viagent_{date.today()}.csv"


# create the task and get the optimal policy
task = TASK_CLASS.create_env(**config["env_kwargs"])
task = CnnWrapper(task)

# create the monitor
task = Monitor(task, LOG_DIR)

pi, _ = task.get_optimal_policy()


### Model + Training Parameters


def make_model():
    agent = AgentClass.make_from_configs(
        task, config["agent_config"], config["vae_config"], config["env_kwargs"]
    )
    return agent


agent = make_model()
total_params = sum(p.numel() for p in agent.state_inference_model.parameters())
print(f"Number of parameters: {total_params}")


agent = make_model()
# agent.learn(2048, estimate_batch=True, progress_bar=True)
agent.learn(10000, estimate_batch=True, progress_bar=True)
