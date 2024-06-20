from stable_baselines3 import PPO as StableBaselinesPPO
from torch import FloatTensor

from model.agents.utils.base_agent import BaseAgent
from model.training.rollout_data import RolloutBuffer


class PPO(StableBaselinesPPO, BaseAgent):
    """
    wrapper for PPO with useful functions
    """

    def update_from_batch(self, batch: RolloutBuffer):
        raise NotImplementedError

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:
        return (
            self.policy.get_distribution(obs.permute(0, 3, 1, 2))
            .distribution.probs.clone()
            .detach()
            .numpy()
        )
