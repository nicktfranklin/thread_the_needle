from dataclasses import dataclass

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor
from torch.utils.data import DataLoader

from state_inference.gridworld_env import ActType, RewType
from state_inference.utils.pytorch_utils import DEVICE, convert_8bit_to_float


@dataclass
class OaroTuple:
    obs: Tensor
    a: ActType
    r: RewType
    obsp: Tensor
    index: int  # unique index for each trial


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


class SoftmaxPolicy:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        """
        Initialize a set of q_values if they don't exist, setting them to
        a default value
        """

        if s not in self.q_values:
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init
            self.q_values[s] = q_init

    def get_distribution(self, s: int) -> Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = torch.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = torch.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)
