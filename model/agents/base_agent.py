from abc import ABC, abstractmethod

import torch
from stable_baselines3.common.vec_env import VecEnv
from torch import FloatTensor

from task.gridworld import ActType


class BaseAgent(ABC):
    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> FloatTensor:
        ...

    @abstractmethod
    def get_env(self) -> VecEnv:
        ...

    def predict_egreedy(self, obs: FloatTensor, epsilon: float) -> ActType:
        """
        Get an epsilon-greedy action from the agent's policy.

        :param obs:
        :param epsilon:
        :return:
        """
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            return self.predict(obs, deterministic=False)

    def collect_rollouts(self, n_rollouts: int, max_steps: int, epsilon: float = 0.0):
        """
        Collect rollouts from the environment using the agent's policy.

        :param env:
        :param n_rollouts:
        :param max_steps:
        :param epsilon:
        :return:
        """
        env = self.get_env()
        rollouts = []
        for _ in range(n_rollouts):
            obs = env.reset()
            rollout = []
            for _ in range(max_steps):
                action = self.predict_egreedy(obs, epsilon)
                next_obs, reward, done, info = env.step(action)
                rollout.append((obs, action, next_obs, reward))
                obs = next_obs
                if done:
                    break
            rollouts.append(rollout)
        return rollouts

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
