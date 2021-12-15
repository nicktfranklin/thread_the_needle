from typing import List, Union

import numpy as np
import scipy.sparse


class ValueIterationNetwork:
    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_actions: int = 4,
        gamma: float = 0.8,
        initialization_noise: float = 0.1,
    ):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.initialization_noise = initialization_noise
        self.n_actions = n_actions
        self.gamma = gamma

        assert (self.gamma > 0) and (
            self.gamma < 1
        ), "gamma must be greater than zero and less than 1"

    def inference(
        self,
        transition_function: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
        reward_function: List[np.ndarray],
        iterations: int = 100,
    ):

        return self.value_iteration(
            transition_function,
            reward_function,
            self.n_rows,
            self.n_columns,
            self.gamma,
            iterations,
            self.initialization_noise,
        )

    @staticmethod
    def value_iteration(
        transition_functions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        gamma: float = 0.8,
        iterations: int = 10,
        initialization_noise: float = 0.1,  # in standard deviation
    ):
        """
        For the reward, transition matrix, list indicies correspond to actions.  E.g.
        the 0th action will have the 0the transition/reward function.

        This assumes a square-grid structure
        """

        tiling = ValueIterationNetwork.define_lattice_tiling(n_rows, n_columns)
        n_states = n_rows * n_columns
        n_actions = len(reward_functions)

        # initialize the value functions
        value_function = ValueIterationNetwork._initialize_value_function(
            n_rows, n_columns, initialization_noise
        )
        value_0 = value_function[tiling]
        value_1 = value_function[~tiling]

        # dynamic programming algorithm
        q_0, q_1 = (
            [np.empty((n_states // 2, n_actions))] * n_actions,
            [np.empty((n_states // 2, n_actions))] * n_actions,
        )
        for ii in range(1, iterations + 1):

            if np.random.rand() > 0.5:
                # calculate tiled values
                for a, (r_a, t_a) in enumerate(
                    zip(reward_functions, transition_functions)
                ):
                    q_0[a] = r_a[tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_1)

                value_0 = np.max(q_0, axis=0)

                for a, (r_a, t_a) in enumerate(
                    zip(reward_functions, transition_functions)
                ):
                    q_1[a] = r_a[~tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_0)

                value_1 = np.max(q_1, axis=0)
            else:
                for a, (r_a, t_a) in enumerate(
                    zip(reward_functions, transition_functions)
                ):
                    q_1[a] = r_a[~tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_0)

                value_1 = np.max(q_1, axis=0)

                # calculate tiled values
                for a, (r_a, t_a) in enumerate(
                    zip(reward_functions, transition_functions)
                ):
                    q_0[a] = r_a[tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_1)

                value_0 = np.max(q_0, axis=0)

        value_function[tiling] = value_0
        value_function[~tiling] = value_1

        state_action_values = np.zeros((n_states, n_actions))
        state_action_values[tiling, :] = np.array(q_0).T
        state_action_values[~tiling, :] = np.array(q_1).T

        return state_action_values, value_function

    @staticmethod
    def define_lattice_tiling(n_rows: int, n_columns: int) -> np.ndarray:
        row_order_a = [ii % 2 for ii in range(n_columns)]
        row_order_b = [ii % 2 for ii in range(1, n_columns + 1)]
        tiling = []
        for ii in range(n_rows):
            if ii % 2 == 0:
                tiling += row_order_a
            else:
                tiling += row_order_b

        return np.array(tiling, dtype=bool)

    @staticmethod
    def _initialize_value_function(
        n_rows: int, n_columns: int, noise: float
    ) -> np.ndarray:
        return np.random.normal(loc=0, scale=noise, size=(n_rows, n_columns)).reshape(
            -1
        )
