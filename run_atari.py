import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

import gymnasium as gym
import torch
import yaml
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from model.agents.lookahead_value_iteration import (
    LookaheadViAgent as ValueIterationAgent,
)
from model.agents.ppo import PPO
from model.training.callbacks import AtariCallback
from model.training.rollout_data import RolloutDataset
from task.gridworld import GridWorldEnv
from utils.config_utils import (
    load_config,
    parse_configs,
    parse_model_config,
    parse_task_config,
)
from utils.pytorch_utils import DEVICE

logging.info(f"python {sys.version}")
logging.info(f"torch {torch.__version__}")
logging.info(f"device = {DEVICE}")


BASE_FILE_NAME = "atari"

### Configuration files
parser = argparse.ArgumentParser()

parser.add_argument("--vae_config", default="configs/vae_config.yml")
parser.add_argument("--task_config", default="configs/env_config.yml")
parser.add_argument("--agent_config", default="configs/agent_config.yml")
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")
parser.add_argument("--results_dir", default=f"simulations/")
parser.add_argument("--log_dir", default=f"logs/{BASE_FILE_NAME}_{date.today()}/")
parser.add_argument("--n_training_samples", default=2_500_000)  # 50000
parser.add_argument("--n_rollout_samples", default=20000)  # 50000
parser.add_argument("--n_batch", default=16)  # 24
parser.add_argument("--atari_env", default="ALE/Pong-v5")
parser.add_argument("--ppo", action="store_true")


@dataclass
class Config:

    log_dir: str
    model_name: str
    results_dir: str
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]
    env_kwargs: Dict[str, Any]

    n_training_samples: int
    n_rollout_samples: int

    n_batch: int
    epsilon: float = 0.02

    atari_env: str = "ALE/Pong-v5"

    use_ppo: bool = False

    @classmethod
    def construct(cls, args: argparse.Namespace):

        return cls(
            log_dir=args.log_dir,
            model_name="ppo" if args.ppo else args.model_name,
            vae_config=parse_model_config(args.model_name, args.vae_config),
            env_kwargs=parse_task_config(args.task_name, args.task_config),
            agent_config=load_config(args.agent_config),
            n_training_samples=int(args.n_training_samples),
            n_rollout_samples=int(args.n_rollout_samples),
            results_dir=args.results_dir,
            n_batch=int(args.n_batch),
            atari_env=args.atari_env,
            use_ppo=args.ppo,
        )


def make_env(configs: Config, batch: int = 0) -> GridWorldEnv:
    # create the task
    task = gym.make(configs.atari_env)

    # create the monitor
    log_name = f"{configs.model_name}_{configs.atari_env.split('/')[-1]}_batch_{batch}"
    task = Monitor(task, os.path.join(configs.log_dir, log_name))

    # prepare for ATARI
    task = AtariWrapper(task, screen_size=64)

    return task


def train_agent(configs: Config, batch: int = 0):

    # create task
    task = make_env(configs, batch=batch)
    task = Monitor(task, configs.log_dir)  # not sure I use this

    callback = [
        AtariCallback(configs.log_dir),
        CheckpointCallback(save_freq=250_000, save_path="checkpoints/"),
    ]

    if configs.use_ppo is False:
        agent = ValueIterationAgent.make_from_configs(
            task, configs.agent_config, configs.vae_config, configs.env_kwargs
        )
    else:
        agent = PPO(
            "CnnPolicy",
            task,
            verbose=0,
            n_steps=min(configs.n_training_samples, 2048),
            batch_size=64,
            n_epochs=10,
        )
    agent.learn(
        total_timesteps=configs.n_training_samples, progress_bar=True, callback=callback
    )

    data = {"rewards": callback.rewards, "evaluations": callback.evaluations}

    return agent, data


def main():
    config = Config.construct(parser.parse_args())

    atari_env = config.atari_env.split("/")[-1]

    file_pattern = f"{config.model_name}_{BASE_FILE_NAME}_{atari_env}_{date.today()}"
    print(f"file_pattern = {file_pattern}")

    config_record = f"logs/{BASE_FILE_NAME}_config_{date.today()}.yaml"
    with open(config_record, "w") as f:
        yaml.dump(config.__dict__, f)

    # Create log dir
    os.makedirs(config.log_dir, exist_ok=True)

    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)

    # train ppo
    batched_data = []
    for ii in range(config.n_batch):
        logging.info(f"running batch {ii}")
        agent, data = train_agent(config, ii)
        data["batch"] = ii
        batched_data.append(data)

        with open(f"{config.results_dir}{file_pattern}_batched_data.pkl", "wb") as f:
            pickle.dump(batched_data, f)

    # Buffer doesn't work, for now
    # rollout_buffer = RolloutDataset()
    # rollout_buffer = agent.collect_buffer(
    #     agent.env, rollout_buffer, n=5000, epsilon=config.epsilon
    # )

    # with open(f"{config.results_dir}{file_pattern}_rollouts.pkl", "wb") as f:
    #     pickle.dump(rollout_buffer, f)


if __name__ == "__main__":
    main()
