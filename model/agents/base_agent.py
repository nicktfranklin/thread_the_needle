from abc import ABC, abstractmethod

import torch
from torch import FloatTensor

from task.gridworld import ActType


class BaseAgent(ABC):
    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> FloatTensor:
        ...

    def get_policy_prob(self, env, n_states: int, map_height: int, cnn=True):
        # reshape to match env standard (HxWxC) -> not standard
        shape = [map_height, map_height]
        if cnn:
            shape = [map_height, map_height, 1]

        obs = [
            torch.tensor(env.env_method("generate_observation", s)[0]).view(*shape)
            for s in range(n_states)
        ]
        obs = torch.stack(obs)
        with torch.no_grad():
            return self.get_pmf(obs)
