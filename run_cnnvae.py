import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

import numpy as np
import torch
import yaml
from stable_baselines3.common.monitor import Monitor

from model.agents.ppo import PPO
from model.agents.value_iteration import ValueIterationAgent as ViAgent
from task.gridworld import CnnWrapper, GridWorldEnv
from task.gridworld import ThreadTheNeedleEnv as Environment
from utils.config_utils import parse_configs
from utils.pytorch_utils import DEVICE, convert_8bit_to_float, make_tensor

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
parser.add_argument("--n_training_samples", default=256)
parser.add_argument("--n_rollout_samples", default=256)
parser.add_argument("--results_path", default="tmp/tmp.json")


@dataclass
class Config:
    log_dir: str
    results_path: str
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
            results_path=args.results_path,
        )


def make_env(configs: Config) -> GridWorldEnv:
    # create the task
    task = CnnWrapper(Environment.create_env(**configs.env_kwargs))

    # create the monitor
    task = Monitor(task, configs.log_dir)

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
    config = Config.construct(parser.parse_args())
    print(yaml.dump(config.__dict__))

    # Create log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # create task
    task = make_env(config)
    task = Monitor(task, config.log_dir)
    pi, _ = task.get_optimal_policy()  # save for analysis

    # train ppo
    # ppo = train_ppo(config, task)

    # rollout_buffer = Buffer()
    # collect_buffer(ppo.policy, task, rollout_buffer)

    ## Model + Training Parameters
    agent = ViAgent.make_from_configs(
        task, config.agent_config, config.vae_config, config.env_kwargs
    )
    # agent.update_from_batch(rollout_buffer, progress_bar=True)

    # # train the VI agent
    agent.learn(
        config.n_training_samples,
        progress_bar=True,
    )

    # save this output
    pmf = agent.get_policy_prob(
        agent.get_env(),
        n_states=config.env_kwargs["n_states"],
        map_height=config.env_kwargs["map_height"],
        cnn=True,
    )

    obs = convert_8bit_to_float(
        torch.stack(
            [
                torch.tensor(task.observation_model(s), dtype=torch.long)
                for s in range(task.transition_model.n_states)
                for _ in range(1)
            ]
        )
    )[:, None, ...].to(DEVICE)
    z = agent.state_inference_model.get_state(obs)
    z = z.dot(agent.hash_vector)  # save this output

    state_rewards = np.array(
        [agent.reward_estimator.get_avg_reward(z0) for z0 in z]
    ).reshape(
        20, 20
    )  # save this output

    state_value_function = np.array(
        [agent.value_function.get(z0, np.nan) for z0 in z]
    ).reshape(20, 20)

    room_1_mask = (np.arange(400) < 200) * (np.arange(400) % 20 < 10)
    room_2_mask = (np.arange(400) >= 200) * (np.arange(400) % 20 < 10)
    room_3_mask = np.arange(400) % 20 >= 10

    output_json = {
        "policy_pmf": pmf,
        "state_embeddings": z,
        "state_rewards": state_rewards,
        "state_values": state_value_function,
        "score": np.sum(pi * pmf, axis=1).mean(),
        "score_room1": np.sum(pi[room_1_mask] * pmf[room_1_mask], axis=1).mean(),
        "score_room2": np.sum(pi[room_2_mask] * pmf[room_2_mask], axis=1).mean(),
        "score_room3": np.sum(pi[room_3_mask] * pmf[room_3_mask], axis=1).mean(),
    }

    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    with open(config.results_path, "w") as f:
        json.dump({k: to_list(v) for k, v in output_json.items()}, f)


if __name__ == "__main__":
    main()


# agent = make_model()
# total_params = sum(p.numel() for p in agent.state_inference_model.parameters())
# print(f"Number of parameters: {total_params}")


# agent = make_model()
# # agent.learn(2048, estimate_batch=True, progress_bar=True)
# agent.learn(10000, estimate_batch=True, progress_bar=True)
