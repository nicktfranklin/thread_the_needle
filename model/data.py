from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from task.gridworld import ActType, ObsType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


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


@dataclass
class OaroTuple:
    obs: ObsType
    a: ActType
    r: float
    obsp: Tensor
    index: int  # unique index for each trial
