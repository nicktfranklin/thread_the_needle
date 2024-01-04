from dataclasses import dataclass
from typing import Any, Dict, SupportsFloat, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from task.gridworld import ActType, ObsType, OutcomeTuple
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


# TODO: Refactor and delete this class.  Mixes data and model
class RolloutBuffer:
    def __init__(self):
        self.cached_obs = list()

    def add(self, item):
        self.cached_obs.append(item)

    def reset(self):
        self.cached_obs = list()

    def len(self):
        return len(self.cached_obs)

    def get_all(self):
        return self.cached_obs

    def get_tensor(self, item="obs"):
        return torch.stack([getattr(o, item) for o in self.cached_obs])

    def get_vae_dataloader(self, batch_size: int):
        obs = self.get_tensor("obs")
        obs = convert_8bit_to_float(obs).to(DEVICE)
        obs = obs.permute(0, 3, 1, 2)  # -> NxCxHxW

        return DataLoader(
            obs,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )


class D4rlDataset:
    def __init__(self) -> None:
        self.action = []
        self.obs = []
        self.next_obs = []
        self.reward = []
        self.terminated = []
        self.truncated = []
        self.info = []

    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple):
        self.action.append(action)
        self.obs.append(obs)
        self.next_obs.append(obs_tuple[0])  # these are sucessor observations
        self.reward.append(obs_tuple[1])
        self.terminated.append(obs_tuple[2])
        self.truncated.append(obs_tuple[3])
        self.info.append(obs_tuple[4])

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL"""

        return {
            "observations": np.stack(self.obs),
            "next_observations": np.stack(self.next_obs),
            "actions": np.stack(self.action),
            "rewards": np.stack(self.reward),
            "terminated": np.stack(self.terminated),
            "timouts": np.stack(self.truncated),  # timeouts are truncated
            "infos": self.info,
        }

    def reset_buffer(self):
        self.action = []
        self.obs = []
        self.next_obs = []
        self.reward = []
        self.terminated = []
        self.truncated = []
        self.info = []


# TODO: Remove this class
@dataclass
class OaroTuple:
    obs: ObsType
    a: ActType
    r: float
    next_obs: Tensor
    index: int  # unique index for each trial
