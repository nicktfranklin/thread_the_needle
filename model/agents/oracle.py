from torch import FloatTensor

from model.agents.base_agent import BaseAgent
from task.gridworld import GridWorldEnv


class Oracle(BaseAgent):
    """uses the optimal policy to collect rollouts"""

    def __init__(self, task: GridWorldEnv, epsilon: float = 0.0) -> None:
        super().__init__(task)
        self.pi = task.get_optimal_policy()
        self.epison = epsilon
        self.task = task

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:
        state = self.task.observation_model.decode_obs(obs)
        return self.pi[state]
