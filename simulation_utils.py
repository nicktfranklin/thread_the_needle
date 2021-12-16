import numpy as np


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


def inverse_cmf_sampler(pmf: np.ndarray) -> int:
    return np.sum(pmf.cumsum() < np.random.uniform(0, 1))
