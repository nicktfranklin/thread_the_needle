from typing import Tuple

import torch
from torch.utils.data import Dataset

from state_inference.gridworld_env import (
    ObservationModel,
    TransitionModel,
    sample_random_walk,
)
from state_inference.utils.pytorch_utils import convert_8bit_to_float, make_tensor


class ObservationDataset(Dataset):
    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n: int = 10000,
        train: bool = True,
    ) -> Dataset:
        if train:
            self.observations = sample_random_walk(
                n, transition_model, observation_model
            )
        else:
            # for test, use the uncorrupted dataset
            self.observations = torch.stack(
                [
                    make_tensor(observation_model.embed_state(s))
                    for s in range(transition_model.n_states)
                ]
            )

        self.observations = convert_8bit_to_float(self.observations)
        self.n = self.observations.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx]

    def __len__(self):
        return self.n


class PomdpDataset(Dataset):
    """Data set contains (o, o') tuples"""

    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n: int = 100000,
        n_chains: int = 1,
    ) -> None:
        super().__init__()
        assert n % n_chains == 0

        x, y = [], []
        chain_len = n // n_chains

        for _ in range(n_chains):
            obs = sample_random_walk(chain_len + 1, transition_model, observation_model)
            x.append(obs[:-1])
            y.append(obs[1:])
        self.x = torch.vstack(x)
        self.y = torch.vstack(y)

        self.n = n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


class TransitionVaeDataset(Dataset):
    def __init__(
        self,
        observations: torch.tensor,
        actions: torch.tensor,
        successor_obs: torch.tensor,
    ):
        super().__init__()
        self.obs = convert_8bit_to_float(observations)
        self.obsp = convert_8bit_to_float(successor_obs)
        self.actions = actions.float()

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.obs[index], self.actions[index], self.obsp[index]

    def __len__(self):
        return len(self.obs)
