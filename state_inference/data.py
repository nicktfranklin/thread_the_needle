from typing import Tuple

import torch
from torch.utils.data import Dataset

from state_inference.gridworld_env import ObservationModel, TransitionModel
from state_inference.sampling_functions import generate_random_walk


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
            obs = generate_random_walk(
                chain_len + 1, transition_model, observation_model
            )
            x.append(obs[:-1])
            y.append(obs[1:])
        self.x = torch.vstack(x)
        self.y = torch.vstack(y)

        self.n = n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n
