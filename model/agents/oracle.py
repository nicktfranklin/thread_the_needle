from torch import FloatTensor, LongTensor

from model.agents.base_agent import BaseAgent
from model.data import D4rlDataset
from task.gridworld import GridWorldEnv
from task.utils import ActType, ObsType
from utils.sampling_functions import inverse_cmf_sampler


class Oracle(BaseAgent):
    """uses the optimal policy to collect rollouts"""

    def __init__(self, task: GridWorldEnv, epsilon: float = 0.0) -> None:
        super().__init__(task)
        self.pi = task.unwrapped.get_optimal_policy()
        self.epison = epsilon
        self.task = task

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:

        obs = obs.squeeze().cpu().numpy()
        state = self.task.unwrapped.observation_model.decode_obs(obs)

        pi = self.pi[state] * (1 - self.epison) + self.epison / self.task.n_states
        return pi

    def predict(
        self,
        obs: ObsType,
        state: FloatTensor | None = None,
        episode_start: bool | None = None,
        deterministic: bool = False,
    ) -> tuple[ActType, FloatTensor | None]:
        pi = self.get_pmf(obs)
        return inverse_cmf_sampler(pi), None

    def update_from_batch(self, batch: D4rlDataset):
        raise NotImplementedError
