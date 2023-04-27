from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from state_inference.env import ObservationModel, TransitionModel


def generate_random_walk(
    length: int,
    transition_model: TransitionModel,
    observation_model: ObservationModel,
    initial_state: Optional[int] = None,
) -> torch.tensor:
    states = transition_model.sample_states(length, "walk", initial_state=initial_state)
    obs = torch.stack([observation_model(s).unsqueeze(0) for s in states])
    return obs


class ObservationDataset(Dataset):
    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n: int = 10000,
        train: bool = True,
    ) -> Dataset:
        if train:
            self.observations = generate_random_walk(
                n, transition_model, observation_model
            )
        else:
            # for test, use the uncorrupted dataset
            self.observations = torch.stack(
                [
                    observation_model.embed_state(s)
                    for s in range(transition_model.n_states)
                ]
            )
        self.n = self.observations.shape[0]
        # self.observations = self.observations.view(self.n, -1)

    def __getitem__(self, idx):
        return self.observations[idx]

    def __len__(self):
        return self.n


class SequenceDataset(Dataset):
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
            obs = generate_random_walk(
                chain_len + 1, transition_model, observation_model
            )
            x.append(obs[:-1])
            y.append(obs[1:])
            print(obs[:-1].shape)
        self.x = torch.stack(x)
        self.y = torch.stack(y)

        self.n = n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n
