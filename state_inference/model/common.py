from dataclasses import dataclass
from random import choice
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor
from tqdm import trange

from state_inference.gridworld_env import ActType, RewType
from state_inference.model.vae import StateVae
from state_inference.utils.pytorch_utils import DEVICE


@dataclass
class OaroTuple:
    obs: Tensor
    a: ActType
    r: RewType
    obsp: Tensor
    index: int  # unique index for each trial


class BaselineCompatibleAgent:
    def __init__(
        self,
        task,
        state_inference_model: StateVae,
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.rollout_buffer = list()

    def get_env(self):
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            return np.array(0), None
        return np.array(np.random.randint(self.task.action_space.n)), None

    def _batch_estimate(
        self, step: int, last_step: bool, progress_bar: Optional[bool] = False
    ) -> None:
        pass

    def _within_batch_update(
        self, obs: OaroTuple, state: Optional[Tensor], state_prev: Optional[Tensor]
    ) -> None:
        pass

    def _init_state(self):
        return None

    def _init_index(self):
        if len(self.rollout_buffer) == 0:
            return 0
        last_obs = self.rollout_buffer[-1]
        return last_obs.index + 1

    def learn(
        self,
        total_timesteps: int,
        estimate_batch: bool = True,
        progress_bar: bool = False,
        **kwargs,
    ) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        episode_start = True
        state_prev = self._init_state()
        idx = self._init_index()

        if progress_bar:
            iterator = trange(total_timesteps, desc="Steps")
        else:
            iterator = range(total_timesteps)
        for step in iterator:
            action, state = self.predict(obs_prev, state_prev, episode_start)
            episode_start = False

            obs, rew, terminated, _, _ = self.task.step(action.item())
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=torch.tensor(obs_prev),
                a=action.item(),
                r=rew,
                obsp=torch.tensor(obs),
                index=idx,
            )

            self._within_batch_update(obs_tuple, state, state_prev)

            self.rollout_buffer.append(obs_tuple)

            obs_prev = obs
            state_prev = state
            if terminated:
                obs_prev = self.task.reset()[0]
                idx += 1
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)

        if estimate_batch:
            self._batch_estimate(
                step, step == total_timesteps - 1, progress_bar=progress_bar
            )


class SoftmaxPolicy:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        if s not in self.q_values:
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init
            self.q_values[s] = q_init

    def get_distribution(self, s: int) -> Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = torch.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = torch.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)
