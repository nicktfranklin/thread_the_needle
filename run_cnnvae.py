import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from model.agents.base_agent import BaseAgent, collect_buffer
from model.agents.ppo import PPO
from model.agents.value_iteration import ValueIterationAgent as ViAgent
from model.data import D4rlDataset as Buffer
from task.gridworld import CnnWrapper, GridWorldEnv
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
parser.add_argument("--n_training_samples", default=200)
parser.add_argument("--n_rollout_samples", default=200)


@dataclass
class Config:
    log_dir: str
    env_kwargs: Dict[str, Any]
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]

    n_training_samples: int
    n_rollout_samples: int

    @classmethod
    def construct(cls, args: argparse.Namespace):
        configs = parse_configs(args)
        return cls(
            log_dir=args.log_dir,
            env_kwargs=configs["env_kwargs"],
            vae_config=configs["vae_config"],
            agent_config=configs["agent_config"],
            n_training_samples=args.n_training_samples,
            n_rollout_samples=args.n_rollout_samples,
        )


def make_env(configs: Config) -> GridWorldEnv:
    # create the task
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))

    # create the monitor
    task = Monitor(task, configs.log_dir)
    # pi, _ = task.get_optimal_policy()

    return task


def train_ppo(configs: Config, task: GridWorldEnv):
    ppo = PPO(
        "CnnPolicy",
        task,
        verbose=0,
        n_steps=min(configs.n_training_samples, 2048),
        batch_size=64,
        n_epochs=10,
    )
    ppo.learn(total_timesteps=configs.n_training_samples, progress_bar=True)

    return ppo


def main():
    configs = Config.construct(parser.parse_args())
    print(yaml.dump(configs.__dict__))

    # Create log dir
    os.makedirs(configs.log_dir, exist_ok=True)

    # create task
    task = make_env(configs)

    # train ppo
    ppo = train_ppo(configs, task)

    rollout_buffer = Buffer()
    collect_buffer(ppo.policy, task, rollout_buffer)

    ## Model + Training Parameters
    agent = ViAgent.make_from_configs(
        task, configs.agent_config, configs.vae_config, configs.env_kwargs
    )
    # agent.update_from_batch(rollout_buffer, progress_bar=True)

    # train the VI agent
    agent.learn(
        configs.n_training_samples,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()


# agent = make_model()
# total_params = sum(p.numel() for p in agent.state_inference_model.parameters())
# print(f"Number of parameters: {total_params}")


# agent = make_model()
# # agent.learn(2048, estimate_batch=True, progress_bar=True)
# agent.learn(10000, estimate_batch=True, progress_bar=True)
