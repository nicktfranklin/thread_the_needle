from abc import ABC, abstractmethod

import torch
from stable_baselines3.common.vec_env import VecEnv
from torch import FloatTensor
from stable_baselines3.common.base_class import BaseAlgorithm
from model.data import D4rlDataset

from task.gridworld import ActType


class BaseAgent(ABC):
    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> FloatTensor:
        ...

    def get_env(self) -> VecEnv:
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)


    @abstractmethod
    def predict(self, obs: FloatTensor, deterministic: bool = True) -> ActType:
        ...

    def collect_rollouts(self, n_rollouts: int, rollout_buffer: D4rlDataset,  max_steps: int,):

        env = self.get_env()

        for _ in range(n_rollouts):
            obs = env.reset()[0]
            rollout = []
            for _ in range(max_steps):
                action = self.predict(obs)
                outcome_tuple = env.step(action)
                rollout.add(obs, action, outcome_tuple)
                
                obs = outcome_tuple[0]
                done = outcome_tuple[2]
                truncated = outcome_tuple[3]


                if done or truncated:
                    break


        return rollout_buffer
    

    def learn(self, total_timesteps: int, progress_bar: bool=False, **kwargs):
        pass

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
