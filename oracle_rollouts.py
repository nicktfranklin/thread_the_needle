import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

import numpy as np
import torch
import yaml
from stable_baselines3.common.monitor import Monitor

from model.agents.base_agent import BaseAgent
from model.agents.oracle import Oracle
from model.agents.ppo import PPO
from model.data import D4rlDataset as Buffer
from task.gridworld import CnnWrapper, GridWorldEnv
from task.gridworld import ThreadTheNeedleEnv as Environment
from utils.config_utils import parse_configs
from utils.pytorch_utils import DEVICE, convert_8bit_to_float

print(f"python {sys.version}")
print(f"torch {torch.__version__}")
print(f"device = {DEVICE}")


BASE_FILE_NAME = "thread_the_needle_cnn_vae"

### Configuration files
parser = argparse.ArgumentParser()

parser.add_argument("--vae_config", default="configs/vae_config.yml")
parser.add_argument("--task_config", default="configs/env_config.yml")
parser.add_argument("--agent_config", default="configs/agent_config.yml")
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")
parser.add_argument("--results_dir", default=f"simulations/")
parser.add_argument("--log_dir", default=f"logs/{BASE_FILE_NAME}_{date.today()}/")
parser.add_argument("--n_rollout_samples", default=10000)


@dataclass
class Config:
    log_dir: str
    results_dir: str
    env_kwargs: Dict[str, Any]
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]

    n_rollout_samples: int
    epsilon: float = 0.02

    @classmethod
    def construct(cls, args: argparse.Namespace):
        configs = parse_configs(args)
        return cls(
            log_dir=args.log_dir,
            env_kwargs=configs["env_kwargs"],
            vae_config=configs["vae_config"],
            agent_config=configs["agent_config"],
            n_rollout_samples=args.n_rollout_samples,
            results_dir=args.results_dir,
        )


def make_env(configs: Config) -> GridWorldEnv:
    # create the task
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))

    # create the monitor
    task = Monitor(task, configs.log_dir)

    return task


def main():
    config = Config.construct(parser.parse_args())

    config_record = f"logs/{BASE_FILE_NAME}_config_{date.today()}.yaml"
    with open(config_record, "w") as f:
        yaml.dump(config.__dict__, f)

    # Create log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # create task
    task = make_env(config)
    task = Monitor(task, config.log_dir)

    # # train ppo``
    oracle = Oracle(task, epsilon=0.1)

    rollout_buffer = Buffer()
    rollout_buffer = oracle.collect_buffer(
        task, rollout_buffer, n=config.n_rollout_samples, epsilon=config.epsilon
    )

    with open(f"{config.results_dir}oracle_rollouts.pkl", "wb") as f:
        pickle.dump(rollout_buffer, f)


if __name__ == "__main__":
    main()


# agent = make_model()
# total_params = sum(p.numel() for p in agent.state_inference_model.parameters())
# print(f"Number of parameters: {total_params}")


# agent = make_model()
# # agent.learn(2048, estimate_batch=True, progress_bar=True)
# agent.learn(10000, estimate_batch=True, progress_bar=True)
