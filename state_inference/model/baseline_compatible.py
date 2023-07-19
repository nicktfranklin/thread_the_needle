from dataclasses import dataclass
from random import choice
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from state_inference.gridworld_env import ActType, ObsType, RewType
from state_inference.model.vae import StateVae
from state_inference.utils.pytorch_utils import (
    convert_8bit_array_to_float_tensor,
    train,
)

BATCH_SIZE = 64
N_EPOCHS = 10000
OPTIM_KWARGS = dict(lr=3e-4)
GRAD_CLIP = True


@dataclass
class OaroTuple:
    obs: ObsType
    a: ActType
    r: RewType
    obsp: ObsType


class RandomAgent:
    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        set_action: Set[ActType],
        optim_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model
        self.set_action = set_action
        self.cached_observations = list()

        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        self.optim = AdamW(self.state_inference_model.parameters(), **optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size

    def get_hashed_state(self, obs: int) -> tuple[int]:
        return tuple(*self.state_inference_model.get_state(obs))

    def predict(self, observation: ObsType) -> tuple[ActType, None]:
        return choice(list(self.set_action)), None

    def update(self, o_prev: ObsType, o: ObsType, a: ActType, r: float) -> None:
        pass

    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        obs = [o.obs for o in self.cached_observations]

        return DataLoader(
            torch.stack(convert_8bit_array_to_float_tensor(obs)),
            batch_size=batch_size,
            shuffle=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        _ = train(self.state_inference_model, dataloader, self.optim, self.grad_clip)

    def get_hashed_state(self, obs: int) -> tuple[int]:
        return tuple(*self.state_inference_model.get_state(obs))

    def learn(self, total_timesteps: int, **kwargs) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        for _ in range(total_timesteps):
            action, _ = self.predict(obs_prev)
            obs, rew, terminated, _, _ = self.task.step(action)

            self.cached_observations.append(
                OaroTuple(obs=obs_prev, a=action, r=rew, obsp=obs)
            )
            assert obs.shape

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
                # print(obs_prev)
            assert obs_prev.shape
            assert not isinstance(obs_prev, tuple)

        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
