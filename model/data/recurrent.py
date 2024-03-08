import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model.data.d4rl import D4rlDataset


class RecurrentDataset(Dataset):
    def __init__(self, buffer: D4rlDataset, seq_len: int):
        self.seq_len = seq_len  # maximum length of a sequence

        ds = buffer.get_dataset()
        self.observations = ds["observations"]

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

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        """return a sequence of observations.
        each sequence starts either idx - seq_len or idx - run_len, whichever is shorter
        """

        run_len = min([self.time_in_run[idx], self.seq_len])

        return torch.from_numpy(self.observations[idx - run_len + 1 : idx + 1, ...])

    def collate_fn(self, batch):
        """
        Pad the batch of sequences to the same length, and return a tensor of shape
        (seq_len, batch_size, *obs_shape)
        """
        return pad_sequence(batch, batch_first=False, padding_value=0)
