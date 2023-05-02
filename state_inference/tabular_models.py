from abc import ABC, abstractmethod
from typing import Dict, Hashable, List, Set, Union

import numpy as np
import torch
from torch import nn

from state_inference.env import OaorTuple


class TabularAgent(ABC):
    def __init__(
        self, observation_model: nn.Module, set_action: Set[Union[int, str]]
    ) -> None:
        self.observation_model = observation_model
        self.set_action = set_action

    @abstractmethod
    def take_action(
        self, observation: Union[torch.Tensor, np.ndarray]
    ) -> Union[str, int]:
        ...

    def update(self, obs_tuple: OaorTuple) -> None:
        



class TabularTransitionModel:
    ## Note: does not take in actions

    def __init__(self):
        self.transitions = dict()
        self.pmf = dict()

    def update(self, s: Hashable, sp: Hashable):
        if s in self.transitions:
            if sp in self.transitions[s]:
                self.transitions[s][sp] += 1
            else:
                self.transitions[s][sp] = 1
        else:
            self.transitions[s] = {sp: 1}

        N = float(sum(self.transitions[s].values()))
        self.pmf[s] = {sp0: v / N for sp0, v in self.transitions[s].items()}

    def batch_update(self, list_states: List[Hashable]):
        for ii in range(len(list_states) - 1):
            self.update(list_states[ii], list_states[ii + 1])

    def get_transition_probs(self, s: Hashable) -> Dict[Hashable, float]:
        # if a state is not in the model, assume it's self-absorbing
        if s not in self.pmf:
            return {s: 1.0}
        return self.pmf[s]


class TabularRewardEstimator:
    def __init__(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():  # pylint: disable=consider-iterating-dictionary
            self.counts[s] += np.array([r, 1])
        else:
            self.counts[s] = np.array([r, 1])

        self.state_reward_function[s] = self.counts[s][0] / self.counts[s][1]

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_reward(self, state):
        return self.state_reward_function[state]


def value_iteration(
    t: Dict[Union[str, int], TabularTransitionModel],
    r: TabularRewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _inner_sum(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * v[sp]
        return _sum

    def _expected_reward(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * r.get_reward(sp)
        return _sum

    for _ in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = _expected_reward(s, a) + gamma * _inner_sum(s, a)
        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v
