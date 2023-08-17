from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
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
        self.observations = self.observations[:, None, ...]
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


class RecurrentVaeDataset(Dataset):
    def __init__(
        self,
        observations: torch.tensor,
        actions: torch.tensor,
        max_sequence_len: int,
    ):
        self.obs = convert_8bit_to_float(observations)
        self.actions = actions.float()
        self.n = max_sequence_len

    def __getitem__(self, index: int) -> Tuple:
        start = max(0, index - self.n)
        return self.obs[start:index], self.actions[start:index], index - start

    def __len__(self):
        return len(self.obs)

    @staticmethod
    def collate_fn(batch):
        obs = [x[0] for x in batch]
        actions = [x[1] for x in batch]
        lengths = [x[2] for x in batch]

        obs = pad_sequence(obs, batch_first=True, padding_value=0)
        actions = pad_sequence(actions, batch_first=True, padding_value=0)

        return (obs, actions), lengths

    @classmethod
    def contruct_dataloader(
        cls,
        observations: torch.tensor,
        actions: torch.tensor,
        max_sequence_len: int = 10,
        batch_size: int = 64,
    ):
        dataset = cls(observations, actions, max_sequence_len)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=cls.collate_fn
        )
