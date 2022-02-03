from typing import Union, List

import numpy as np
import scipy.sparse

import environments


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


def evaluate_policy(policy: np.ndarray, optimal_policy: np.ndarray,) -> np.ndarray:
    """

    :param policy: binary vector of state-action, 1 if in optimal policy, zero otherwise (not pmf)
    :param optimal_policy: pmf
    :return: vector of errors, 1-per state
    """

    return 1 - np.sum(optimal_policy * policy, axis=1)


def get_optimal_sr_from_transitions(
    transition_functions, optimal_policy, n_rows, n_columns, gamma=0.95,
):

    # turn optimal policy in to pmf
    pi = optimal_policy * np.tile(1 / optimal_policy.sum(axis=1), (4, 1)).T

    # marginalize out the optimal policy
    T_sas = np.transpose(transition_functions, (1, 0, 2))
    T_ss = np.array([T_sas[ii, :].T.dot(pi[ii]) for ii in range(pi.shape[0])])

    # calculate the SR
    M = np.linalg.inv(np.eye(n_rows * n_columns) - gamma * T_ss)
    return M


def _check_valid(pos, max_pos):
    return (pos > -1) and (pos < max_pos)


def _get_neighbors(state: int, n_columns: int, n_rows: int) -> List[int]:
    r, c = environments.get_position_from_state(state, n_columns)
    neighbors = []
    for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        if _check_valid(r + dr, n_rows) and _check_valid(c + dc, n_columns):
            neighbors.append(
                environments.get_state_from_position(r + dr, c + dc, n_columns)
            )
    return neighbors


def find_shortest_path(
    state_value_function: np.ndarray,
    goal_state: int,
    start_state: int,
    n_columns: int,
    n_rows: int,
) -> np.ndarray:
    state = start_state
    path = []
    while state is not goal_state:
        neighbors = _get_neighbors(state, n_columns, n_rows)
        values = {n: state_value_function[n] for n in neighbors if n not in path}
        state = max(values, key=values.get)
        path.append(state)

    return path


def find_sortest_path_length(
    state_value_function: np.ndarray,
    goal_state: int,
    start_state: int,
    n_columns: int,
    n_rows: int,
) -> int:
    path = find_shortest_path(
        state_value_function, goal_state, start_state, n_columns, n_rows
    )
    return len(path)
