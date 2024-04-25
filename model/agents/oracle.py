import numpy as np
from torch import FloatTensor

from model.agents.utils.base_agent import BaseAgent
from model.training.rollout_data import RolloutDataset
from task.gridworld import GridWorldEnv
from task.utils import ActType, ObsType
from utils.sampling_functions import inverse_cmf_sampler


class Oracle(BaseAgent):
    """uses the optimal policy to collect rollouts"""

    def __init__(self, task: GridWorldEnv, epsilon: float = 0.0) -> None:
        super().__init__(task)
        self.task: GridWorldEnv = task.unwrapped
        self.pi, self.state_values = self.task.get_optimal_policy()
        self.epison = epsilon

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:

        obs = obs.squeeze().cpu().numpy()
        state = self.task.observation_model.decode_obs(obs)

        # normalize to a pdf
        pi = self.pi[state] / self.pi[state].sum()
        # e-greedy
        pi = pi * (1 - self.epison) + self.epison * np.ones_like(pi) / len(pi)
        return pi

    def predict(
        self,
        obs: ObsType,
        state: FloatTensor | None = None,
        episode_start: bool | None = None,
        deterministic: bool = False,
    ) -> tuple[ActType, FloatTensor | None]:
        pi = self.get_pmf(obs)
        print(pi, pi.cumsum())
        return inverse_cmf_sampler(pi), None

    def update_from_batch(self, batch: RolloutDataset):
        raise NotImplementedError
