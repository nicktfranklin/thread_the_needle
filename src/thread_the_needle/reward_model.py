from typing import Dict

import numpy as np

from .utils import RewType, StateType


class RewardModel:
    def __init__(
        self,
        successor_state_rew: Dict[StateType, RewType],
        movement_penalty: float = 0.0,
    ) -> None:
        self.successor_state_rew = successor_state_rew
        self.movement_penalty = movement_penalty

    def get_reward(self, state: StateType):
        return self.successor_state_rew.get(state, self.movement_penalty)

    def get_rew_range(self) -> tuple[RewType, RewType]:
        rews = self.successor_state_rew.values()
        return tuple([min(rews), max(rews)])

    def construct_rew_func(self, transition_function: np.ndarray) -> np.ndarray:
        """
        transition_function -> [n_actions, n_states, n_states]
        returns: a reward function with dimensions [n_states, n_actions]
        """
        _, n_s, _ = transition_function.shape
        reward_function = np.zeros(n_s)
        for sp in range(n_s):
            reward_function[sp] = self.get_reward(sp)
        return np.matmul(transition_function, reward_function)
