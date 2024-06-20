from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import Tensor

from task.gridworld import ActType, ObsType, OutcomeTuple

ObservationTuple = namedtuple("ObservationTuple", "obs a r next_obs")


class RolloutDataset:
    """
    This class is meant to be consistent with the dataset in d4RL
    """

    def __init__(
        self,
        action: Optional[List[ActType]] = None,
        obs: Optional[List[ObsType]] = None,
        next_obs: Optional[List[ObsType]] = None,
        reward: Optional[List[float]] = None,
        terminated: Optional[List[bool]] = None,
        truncated: Optional[List[bool]] = None,
        info: Optional[List[Dict[str, Any]]] = None,
        capacity: Optional[int] = None,
    ) -> None:
        self.action = action if action is not None else []
        self.obs = obs if obs is not None else []
        self.next_obs = next_obs if next_obs is not None else []
        self.reward = reward if reward is not None else []
        self.terminated = terminated if terminated is not None else []
        self.truncated = truncated if truncated is not None else []
        self.info = info if info is not None else []
        self.capcity = capacity

    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple):
        self.action.append(action)
        self.obs.append(obs)
        self.next_obs.append(obs_tuple[0])  # these are sucessor observations
        self.reward.append(obs_tuple[1])
        self.terminated.append(obs_tuple[2])
        self.truncated.append(obs_tuple[3])
        self.info.append(obs_tuple[4])

        if self.capcity is not None and len(self.obs) > self.capcity:
            self.action.pop(0)
            self.obs.pop(0)
            self.next_obs.pop(0)
            self.reward.pop(0)
            self.terminated.pop(0)
            self.truncated.pop(0)
            self.info.pop(0)

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL"""

        return {
            "observations": np.stack(self.obs),
            "next_observations": np.stack(self.next_obs),
            "actions": np.stack(self.action),
            "rewards": np.stack(self.reward),
            "terminated": np.stack(self.terminated),
            "timouts": np.stack(self.truncated),  # timeouts are truncated
            "infos": self.info,
        }

    def get_obs(self, idx: int) -> ObservationTuple:
        return ObservationTuple(
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
        )

    def __len__(self) -> int:
        return len(self.obs)

    def reset_buffer(self):
        self.action = []
        self.obs = []
        self.next_obs = []
        self.reward = []
        self.terminated = []
        self.truncated = []
        self.info = []
