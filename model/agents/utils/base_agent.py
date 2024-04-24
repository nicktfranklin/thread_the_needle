import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from torch import FloatTensor
from tqdm import tqdm, trange

from model.data.d4rl import D4rlDataset
from task.gridworld import ActType, GridWorldEnv, ObsType
from utils.sampling_functions import inverse_cmf_sampler


class BaseAgent(ABC):
    def __init__(self, task: GridWorldEnv) -> None:
        super().__init__()
        self.task = task

    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> FloatTensor: ...

    def get_env(self) -> VecEnv:
        ## used for compatibility with stablebaseline code, use with caution
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    @abstractmethod
    def predict(
        self,
        obs: ObsType,
        state: Optional[FloatTensor] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False,
    ) -> tuple[ActType, Optional[FloatTensor]]: ...

    def _init_state(self) -> Optional[FloatTensor]:
        return None

    def update_rollout_policy(self, rollout_buffer: D4rlDataset) -> None:
        pass

    def collect_rollouts(
        self,
        n_rollout_steps: int,
        rollout_buffer: D4rlDataset,
        progress_bar: Optional[Iterable] = None,
    ):
        task = self.task
        obs = task.reset()[0]
        state = self._init_state()
        episode_start = True
        for _ in range(n_rollout_steps):
            action, state = self.predict(obs, state, episode_start, deterministic=False)
            episode_start = False

            outcome_tuple = task.step(action)
            rollout_buffer.add(obs, action, outcome_tuple)

            self.update_rollout_policy(rollout_buffer)

            # get the next obs from the observation tuple
            obs, _, done, truncated, _ = outcome_tuple

            if done or truncated:
                obs = task.reset()[0]
            if progress_bar is not None:
                progress_bar.update(1)

        return rollout_buffer

    @abstractmethod
    def update_from_batch(self, batch: D4rlDataset): ...

    def learn(self, total_timesteps: int, progress_bar: bool = False, **kwargs):
        logging.info("Calling Library learn method")
        if progress_bar is not None:
            progress_bar = trange(total_timesteps, position=0, leave=True)

        self.rollout_buffer = D4rlDataset()

        # alternate between collecting rollouts and batch updates
        n_rollout_steps = self.n_steps if self.n_steps is not None else total_timesteps

        num_timesteps = 0
        while num_timesteps < total_timesteps:
            self.rollout_buffer.reset_buffer()
            if progress_bar is not None:
                progress_bar.set_description("Collecting Rollouts")

            self.rollout_buffer = self.collect_rollouts(
                n_rollout_steps, self.rollout_buffer, progress_bar=progress_bar
            )
            num_timesteps += n_rollout_steps

            if progress_bar is not None:
                progress_bar.set_description("Updating Batch")

            self.update_from_batch(self.rollout_buffer, progress_bar=True)

        if progress_bar is not None:
            progress_bar.close()

    def get_policy_prob(
        self, env, n_states: int, map_height: int, cnn=True
    ) -> FloatTensor:
        """
        Wrapper for getting the policy probability for each state in the environment.
        Requires a gridworld environment, and samples an observation from each state.

        Returns a tensor of shape (n_states, n_actions)

        :param env:
            :param n_states:
            :param map_height:
        """

        # reshape to match env standard (HxWxC) -> not standard
        shape = [map_height, map_height]
        if cnn:
            shape = [map_height, map_height, 1]

        obs = [
            torch.tensor(env.env_method("generate_observation", s)[0]).view(*shape)
            for s in range(n_states)
        ]
        obs = torch.stack(obs)
        with torch.no_grad():
            return self.get_pmf(obs)

    # todo: implement a progressbar, max steps, etc.  Should be an inplace method
    # look to the BaseAgent method for inspiration
    def collect_buffer(
        self,
        task: GridWorldEnv,
        buffer: D4rlDataset,
        n: int,
        epsilon: float = 0.05,
    ):
        # collect data
        obs = task.reset()[0]
        done = False

        for _ in tqdm(range(n), desc="Collection rollouts"):
            action_pmf = self.get_pmf(torch.tensor(obs).unsqueeze(0))

            # epsilon greedy
            action_pmf = (1 - epsilon) * action_pmf + epsilon * np.ones_like(
                action_pmf
            ) / len(action_pmf)

            # sample
            action = inverse_cmf_sampler(action_pmf)

            outcome_tuple = task.step(action)
            buffer.add(obs, action, outcome_tuple)

            obs = outcome_tuple[0]
            done = outcome_tuple[2]

            if done:
                obs = task.reset()[0]

        return buffer
