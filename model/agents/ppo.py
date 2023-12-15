from stable_baselines3 import PPO as StableBaselinesPPO
from torch import FloatTensor

from model.agents.base_agent import BaseAgent


class PPO(StableBaselinesPPO, BaseAgent):
    """
    wrapper for PPO with useful functions
    """

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:
        return (
            self.policy.get_distribution(obs.permute(0, 3, 1, 2))
            .distribution.probs.clone()
            .detach()
            .numpy()
        )
