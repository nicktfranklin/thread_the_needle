from typing import Union

import numpy as np
import scipy


def sample_trajectory_until_goal(
    start_state: int, goal_state: int, policy: np.ndarray, transition_functions
):
    current_state = start_state
    state_trajectory = [current_state]
    while current_state != goal_state:
        sampled_action = inverse_cmf_sampler(policy[current_state, :])

        t = transition_functions[sampled_action]

        current_state = inverse_cmf_sampler(t[current_state, :])
        state_trajectory.append(current_state)

    return state_trajectory


def inverse_cmf_sampler(pmf: Union[np.ndarray, scipy.sparse.csr_matrix]) -> int:
    if type(pmf) == scipy.sparse.csr_matrix:
        pmf = pmf.toarray()

    return np.array(np.cumsum(np.array(pmf)) < np.random.rand(), dtype=int).sum()
