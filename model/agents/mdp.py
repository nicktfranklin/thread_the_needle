from random import choices
from typing import Dict, Hashable, List

import numpy as np

from task.utils import ActType
from utils.sampling_functions import inverse_cmf_sampler


class TabularTransitionEstimator:
    ## Note: does not take in actions

    def __init__(self):
        self.transitions = {}
        self.pmf = {}

    def reset(self):
        self.transitions = {}
        self.pmf = {}

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

    def sample(self, state: Hashable) -> Hashable:
        pmf = self.get_transition_probs(state)

        sucessor_states = list(pmf.keys())
        pmf = np.array(list(pmf.values()))

        idx = inverse_cmf_sampler(pmf)

        return sucessor_states[idx]


class TabularStateActionTransitionEstimator:
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions
        self.models = {a: TabularTransitionEstimator() for a in range(n_actions)}
        self.set_states = set()

    def reset(self):
        for m in self.models.values():
            m.reset()
        self.set_states = set()

    def update(self, s: Hashable, a: ActType, sp: Hashable) -> None:
        self.models[a].update(s, sp)
        self.set_states.add(s)

    def get_transition_functions(self):
        return self.models

    def sample(self, s: Hashable, a: ActType) -> None:
        return self.models[a].sample(s)


class TabularRewardEstimator:
    def __init__(self):
        self.counts = {}
        self.state_reward_function = {}

    def reset(self):
        self.counts = {}
        self.state_reward_function = {}

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():  # pylint: disable=consider-iterating-dictionary
            n = self.counts[s].get(float(r), 0)
            self.counts[s][float(r)] = n + 1
        else:
            self.counts[s] = {float(r): 1.0}

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_avg_reward(self, state):
        # get the weighted average
        reward_dict = self.counts.get(state, None)
        if reward_dict is None:
            return np.nan
        r = np.array(list(self.counts[state].keys()))
        n = np.array(list(self.counts[state].values()))
        vs = r @ n / n.sum()
        return vs

    def sample(self, state):
        r = np.array(list(self.counts[state].keys()))
        n = np.array(list(self.counts[state].values()))
        return choices(r, n / n.sum())[0]


def value_iteration(
    t: Dict[ActType, TabularTransitionEstimator],
    r: TabularRewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _successor_value(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * v[sp]
        return _sum

    def _expected_reward(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * r.get_avg_reward(sp)
        return _sum

    for _ in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = _expected_reward(s, a) + gamma * _successor_value(s, a)
        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v
