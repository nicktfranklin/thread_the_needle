from typing import Any, Dict

from torch.utils.data import Dataset


class ViDataset(Dataset):
    def __init__(self, dataset: Dict[str, Any]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["observations"])

    def __getitem__(self, idx):
        return {
            "observations": self.dataset["observations"][idx],
            "next_observations": self.dataset["next_observations"][idx],
            "actions": self.dataset["actions"][idx],
            "rewards": self.dataset["rewards"][idx],
        }
