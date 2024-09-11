import torch


class PpoDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data["observations"])

    def collate_fn(batch):
        return {
            "observations": torch.stack([item["observations"] for item in batch]),
            "next_observations": torch.stack(
                [item["next_observations"] for item in batch]
            ),
            "rewards_to_go": torch.cat([item["rewards_to_go"] for item in batch]),
            "advantages": torch.cat([item["advantages"] for item in batch]),
            "actions": torch.cat([item["actions"] for item in batch]),
            "log_probs": torch.cat([item["log_probs"] for item in batch]),
        }
