import argparse
import logging
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

from model.agents import SbDiscretePpo
from model.training.callbacks import ThreadTheNeedleCallback
from model.training.rollout_data import RolloutBuffer as Buffer
from model.training.scoring import score_model
from task.gridworld import CnnWrapper, GridWorldEnv
from task.gridworld import ThreadTheNeedleEnv as Environment
from utils.config_utils import parse_configs
from utils.pytorch_utils import DEVICE

logging.info(f"python {sys.version}")
logging.info(f"torch {torch.__version__}")
logging.info(f"device = {DEVICE}")


BASE_FILE_NAME = "thread_the_needle_sb_dppo"
MODEL_NAME = "sb_dppo"
TENSORBOARD_PATH = "tensorboard/sb3_log/"


### Configuration files
parser = argparse.ArgumentParser()

parser.add_argument("--vae_config", default="configs/vae_config.yml")
parser.add_argument("--task_config", default="configs/env_config.yml")
parser.add_argument("--agent_config", default="configs/discrete_ppo_config.yml")
parser.add_argument("--task_name", default="thread_the_needle")
parser.add_argument("--model_name", default="cnn_vae")
parser.add_argument("--results_dir", default=f"simulations/")
parser.add_argument("--log_dir", default=f"logs/{BASE_FILE_NAME}_{date.today()}/")
parser.add_argument("--n_training_samples", default=2048 * 25)
parser.add_argument("--n_rollout_samples", default=10000)
parser.add_argument("--n_batch", default=8)


@dataclass
class Config:
    log_dir: str
    results_dir: str
    env_kwargs: Dict[str, Any]
    agent_config: Dict[str, Any]
    vae_config: Dict[str, Any]

    n_training_samples: int
    n_rollout_samples: int

    n_batch: int
    epsilon: float = 0.02

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
            results_dir=args.results_dir,
            n_batch=args.n_batch,
        )


def make_env(configs: Config) -> GridWorldEnv:
    # create the task
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))

    # create the monitor
    task = Monitor(task, configs.log_dir)

    return task


def train_ppo(configs: Config):

    # create task
    task = make_env(configs)
    task = Monitor(task, configs.log_dir)  # not sure I use this

    callback = ThreadTheNeedleCallback()

    ppo = SbDiscretePpo(
        "CnnPolicy",
        task,
        verbose=1,
        n_steps=min(configs.n_training_samples, 2048),
        batch_size=64,
        n_epochs=10,
        tensorboard_log=TENSORBOARD_PATH,
        policy_kwargs=dict(
            features_extractor_kwargs=dict(tau=0.05, z_dim=8, z_layers=8)
        ),
        # policy_kwargs=dict(
        #     features_extractor_kwargs=dict(tau=0.1, z_dim=32, z_layers=16)
        # ),
        # device=DEVICE,
    )
    # ppo.to(DEVICE)
    ppo.learn(
        total_timesteps=configs.n_training_samples,
        progress_bar=True,
        callback=callback,
    )

    data = {"rewards": callback.rewards, "evaluations": callback.evaluations}

    return ppo, data


def main():
    config = Config.construct(parser.parse_args())

    config_record = f"logs/{BASE_FILE_NAME}_config_{date.today()}.yaml"
    with open(config_record, "w") as f:
        yaml.dump(config.__dict__, f)

    # Create log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # train ppo
    batched_data = []
    for ii in range(config.n_batch):
        logging.info(f"running batch {ii}")
        ppo, data = train_ppo(config)
        data["batch"] = ii
        batched_data.append(data)

    rollout_buffer = Buffer()
    rollout_buffer = ppo.collect_buffer(
        ppo.env.envs[0], rollout_buffer, n=1000, epsilon=config.epsilon
    )

    with open(
        f"{config.results_dir}{MODEL_NAME}_rollouts_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(rollout_buffer, f)

    with open(
        f"{config.results_dir}{MODEL_NAME}_batched_data_{date.today()}.pkl", "wb"
    ) as f:
        pickle.dump(batched_data, f)


if __name__ == "__main__":
    main()
