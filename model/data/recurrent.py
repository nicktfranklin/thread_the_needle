import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model.data.rollout_data import RolloutDataset


class RecurrentDataset(Dataset):
    action_pad_value = -1

    def __init__(self, buffer: RolloutDataset, seq_len: int):
        self.seq_len = seq_len  # maximum length of a sequence

        ds = buffer.get_dataset()

        # change to (n_samples, n_channels, height, width)
        self.observations = np.transpose(ds["observations"], (0, 3, 1, 2))

        last_obs_in_run = np.sort(
            np.concatenate(
                [
                    np.argwhere(ds["terminated"]).reshape(-1),
                    np.argwhere(ds["timouts"]).reshape(-1),
                ]
            )
        )

        # add the last observation as an end-point determinsitically
        if len(ds["terminated"]) - 1 not in last_obs_in_run:
            last_obs_in_run = np.concatenate(
                [last_obs_in_run, [len(ds["terminated"]) - 1]]
            )

        self.n_runs = len(last_obs_in_run)
        self.run_lens = np.concatenate(
            [
                [last_obs_in_run[0] + 1],  # first run goes from 0 to last_obs_in_run[0]
                last_obs_in_run[1:] - last_obs_in_run[:-1],
            ]
        )

        repeated_end_points = np.repeat(last_obs_in_run, self.run_lens)

        # Create a sequence of numbers from 0 to the total sum of run_lengths
        sequence = np.arange(np.sum(self.run_lens))

        # Subtract the repeated_end_points from the sequence
        self.obs_remaining_in_run = sequence - repeated_end_points

        # Count the time in run, 1-indexing, as this is a length of run measure
        repeated_run_lengths = np.repeat(self.run_lens, self.run_lens)
        self.time_in_run = repeated_run_lengths + self.obs_remaining_in_run

        self.observations = torch.from_numpy(self.observations)
        self.actions = torch.from_numpy(ds["actions"])
        self.rewards = torch.from_numpy(ds["rewards"])

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx) -> dict[torch.Tensor]:
        """
        Returns a dictionary with the following element:
            - Observations (len t)
            - Actions (len t - 1)
            - Rewards (len t - 1)

        at each time-step, there is an (obs, action, reward, succesor_obs tuple)
        Their causal order is: obs -> action -> reward + succesor_obs.

        Hense, the sequence of obs is one longer than actions or rewards.  We simplify
        the causal structure to

        (obs, action) -> (succesor_obs, reward).  Thus, actions and observations are
        aligned from i = 0, ... , t-1 and rewards and observations are alligned from
        i = 1, ..., t

        return: Dict[str, torch.Tensor]
        """

        # input observation is at time idx, successor observation at time idx+1
        run_len = min([self.time_in_run[idx], self.seq_len])

        # align the sequence start to the (obs, action)
        start = idx - run_len + 1

        # align the end to the last (obs, action), shifted by one for range indexing
        end = idx + 1

        output = {
            "obs": self.observations[start : end + 1, ...],
            "action": self.actions[start:end],
            "reward": self.rewards[start + 1 : end + 1, ...],
        }

        return output

    def collate_fn(self, batch):
        """
        Pad the batch of sequences to the same length, and return a tensor of shape
        (seq_len, batch_size, *obs_shape)
        """

        return {
            "obs": pad_sequence(
                [b["obs"] for b in batch], batch_first=False, padding_value=0
            ),
            "action": pad_sequence(
                [b["action"] for b in batch],
                batch_first=False,
                padding_value=self.action_pad_value,
            ),
            "reward": pad_sequence(
                [b["reward"] for b in batch], batch_first=False, padding_value=0
            ),
        }
