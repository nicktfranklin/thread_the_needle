from argparse import Namespace
from typing import Any, Dict
import yaml


def load_config(file_name: str) -> Dict[str, Any]:
    with open(file_name) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def parse_model_config(model_name: str, config_file: str) -> Dict[str, Any]:
    config = load_config(config_file)
    return config[model_name]


def parse_task_config(task_name: str, config_file: str) -> Dict[str, Any]:
    config = load_config(config_file)
    return config[task_name]["env_kwargs"]


def parse_configs(args: Namespace):
    env_kwargs = parse_task_config(args.task_name, args.task_config)
    vae_config = parse_model_config(args.model_name, args.vae_config)
    agent_config = load_config(args.agent_config)

    return {
        "model_name": args.model_name,
        "env_kwargs": env_kwargs,
        "vae_config": vae_config,
        "agent_config": agent_config,
    }
