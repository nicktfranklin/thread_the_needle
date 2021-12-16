from typing import List, Union

import numpy as np
import scipy.sparse


def define_valid_lattice_transitions(
    n_rows: int, n_columns: int, self_transitions: bool = False
) -> np.ndarray:
    """
    Defines a random transition matrix over a square lattice grid where transitions
    are valid only between neighboring states
    """

    n_states = n_rows * n_columns

    up_transitions = np.concatenate(
        [np.zeros(n_columns), np.ones(n_columns * (n_rows - 1))]
    ).reshape(-1, 1)

    left_transitions = np.concatenate(
        [np.zeros(1), np.ones(n_columns - 1)] * n_rows
    ).reshape(-1, 1)

    self_transitions = np.ones((n_states, 1)) * int(self_transitions)

    right_transitions = np.concatenate(
        [np.ones(n_columns - 1), np.zeros(1)] * n_rows
    ).reshape(-1, 1)

    down_transitions = np.concatenate(
        [np.ones(n_columns * (n_rows - 1)), np.zeros(n_columns)]
    ).reshape(-1, 1)

    return np.concatenate(
        [
            up_transitions,
            left_transitions,
            self_transitions,
            right_transitions,
            down_transitions,
        ],
        axis=1,
    )


def make_diagonal_matrix(
    diagonals: np.ndarray, k: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:
    if sparse:
        return scipy.sparse.diags(diagonals, offsets=k)
    return np.diag(diagonals, k=k)


def convert_movements_to_transition_matrix(
    movements: np.ndarray, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:

    transition_matrix = make_diagonal_matrix(movements[:, 2], k=0, sparse=sparse)
    transition_matrix += make_diagonal_matrix(movements[:-1, 3], k=1, sparse=sparse)
    transition_matrix += make_diagonal_matrix(movements[1:, 1], k=-1, sparse=sparse)

    n_cols = np.sum(movements[:, 0] == 0.0)
    transition_matrix += make_diagonal_matrix(
        movements[n_cols:, 0], k=-n_cols, sparse=sparse
    )
    transition_matrix += make_diagonal_matrix(
        movements[:-n_cols, 4], k=n_cols, sparse=sparse
    )
    return transition_matrix


def draw_dirichlet(valid_movements: np.ndarray) -> np.ndarray:
    _p = np.random.dirichlet(np.ones(np.sum(valid_movements)))
    p = np.zeros_like(valid_movements, dtype=float)
    t = 0
    for ii, isValid in enumerate(valid_movements):
        if isValid:
            p[ii] = _p[t]
            t += 1
    return p


def draw_dirichlet_transitions(movements: np.ndarray) -> np.ndarray:
    return np.array(list(map(draw_dirichlet, np.array(movements, dtype=bool))))


def make_diffusion_transition(movements: np.ndarray) -> np.ndarray:
    def _normalize(x):
        return x / np.sum(x)

    return np.array(list(map(_normalize, movements)))


def sample_random_transition_matrix(
    n_rows: int, n_columns: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)
    movement_probs = draw_dirichlet_transitions(movements)
    return convert_movements_to_transition_matrix(movement_probs, sparse=sparse)


def make_diffision_transition_matrix(
    n_rows: int, n_columns: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)
    movement_probs = make_diffusion_transition(movements)
    return convert_movements_to_transition_matrix(movement_probs, sparse=sparse)


def make_cardinal_movements_prob(
    n_rows: int, n_columns: int, slip_probability: float = 0.05,
) -> List[np.ndarray]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)
    diffused_movements = make_diffusion_transition(movements)

    def _map_movement(x, index):
        if np.isclose(x[index], 0):
            return x

        output = np.array(x) * slip_probability
        output[index] += 1 - slip_probability
        return list(output)

    return [
        np.array(list(map(lambda x: _map_movement(x, ii), diffused_movements)))
        for ii in [0, 1, 3, 4]
    ]


def make_cardinal_transition_matrix(
    n_rows: int, n_columns: int, slip_probability: float = 0.05, sparse: bool = False,
) -> List[Union[np.ndarray, scipy.sparse.csr.csr_matrix]]:

    cardinal_movements = make_cardinal_movements_prob(
        n_rows, n_columns, slip_probability
    )
    return [
        convert_movements_to_transition_matrix(move_probs, sparse=sparse)
        for move_probs in cardinal_movements
    ]


def get_state_action_reward_from_sucessor_rewards(
    reward_function_over_sucessors: np.ndarray,
    transitions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
) -> List[Union[np.ndarray, scipy.sparse.csr_matrix]]:
    reward_function_over_sa = [
        t.dot(reward_function_over_sucessors) for t in transitions
    ]
    return reward_function_over_sa


