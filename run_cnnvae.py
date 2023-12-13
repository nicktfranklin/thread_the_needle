import argparse
from dataclasses import dataclass
import os
import sys
from datetime import date
from typing import Any, Dict

import torch
from stable_baselines3.common.monitor import Monitor

from model.agents.value_iteration import ValueIterationAgent as Agent
from task.gridworld import CnnWrapper
from task.gridworld import ThreadTheNeedleEnv as Environment
from utils.config_utils import parse_configs
from utils.pytorch_utils import DEVICE

print(f"python {sys.version}")
print(f"torch {torch.__version__}")
print(f"device = {DEVICE}")


### Configuration files
parser = argparse.ArgumentParser()

parser.add_argument("--vae_config", default="configs/vae_config.yml")
parser.add_argument("--task_config", default="configs/env_config.yml")
parser.add_argument("--agent_config", default="configs/agent_config.yml")
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")
parser.add_argument(
    "--save_file", default=f"simulations/thread_the_needle_vi_agent_{date.today()}.csv"
)
parser.add_argument("--log_dir", default="logs/")


@dataclass
class Config:
    log_dir: str
    env_kwargs: Dict[str, Any]
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]

    @classmethod
    def construct(cls, args: argparse.Namespace):
        configs = parse_configs(args)
        return cls(
            log_dir=args.log_dir,
            env_kwargs=configs["env_kwargs"],
            vae_config=configs["vae_config"],
            agent_config=configs["agent_config"],
        )


def make_env(configs: Config):
    # create the task and get the optimal policy
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))
    pi, _ = task.get_optimal_policy()

    # create the task and get the optimal policy
    task = Environment.create_env(**configs.env_kwargs)
    task = CnnWrapper(task)

    # create the monitor
    task = Monitor(task, configs.log_dir)
    # pi, _ = task.get_optimal_policy()

    return task


def main():
    configs = Config.construct(parser.parse_args())

    # Create log dir
    os.makedirs(configs.log_dir, exist_ok=True)

    ### Model + Training Parameters
    task = make_env(configs)
    agent = Agent.make_from_configs(
        task, configs.agent_config, configs.vae_config, configs.env_kwargs
    )
    print(agent.state_inference_model)


if __name__ == "__main__":
    main()


# agent = make_model()
# total_params = sum(p.numel() for p in agent.state_inference_model.parameters())
# print(f"Number of parameters: {total_params}")


# agent = make_model()
# # agent.learn(2048, estimate_batch=True, progress_bar=True)
# agent.learn(10000, estimate_batch=True, progress_bar=True)
