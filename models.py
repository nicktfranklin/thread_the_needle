from abc import ABC
from random import choice
from typing import List, Tuple, Union, Type

import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from tqdm import tnrange

from monte_carlo_tree_search import Node
from simulation_utils import inverse_cmf_sampler

# TransitionMatrix: List[Union[np.ndarray, sparse.csr_matrix]]


class PlanningModel:
    def __init__(
        self,
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        n_actions: int = 4,
        gamma: float = 0.8,
        beta: float = 1.0,
    ):
        self.transition_functions = transition_functions
        self.reward_functions = reward_functions
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_actions = n_actions
        self.gamma = gamma
        self.beta = beta

        assert (self.gamma > 0) and (
            self.gamma < 1
        ), "gamma must be greater than zero and less than 1"

        assert beta > 0, "Beta must be strictly positive!"

    def inference(self):
        raise NotImplementedError

    @staticmethod
    def softmax(state_action_values: np.ndarray, beta: float = 1) -> np.ndarray:
        assert beta > 0, "Beta must be strictly positive!"

        def _internal_softmax(q: np.ndarray) -> np.ndarray:
            return np.exp(beta * q - logsumexp(beta * q))

        return np.array(list(map(_internal_softmax, state_action_values)))


class ValueIterationNetwork(PlanningModel):
    def __init__(
        self,
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        n_actions: int = 4,
        gamma: float = 0.8,
        beta: float = 1.0,
        initialization_noise: float = 0.01,
    ):
        PlanningModel.__init__(
            self,
            transition_functions,
            reward_functions,
            n_rows,
            n_columns,
            n_actions,
            gamma,
            beta,
        )
        self.initialization_noise = initialization_noise

    def inference(self, iterations: int = 100):

        state_action_values, _ = self.value_iteration(
            self.transition_functions,
            self.reward_functions,
            self.n_rows,
            self.n_columns,
            self.gamma,
            iterations,
            self.initialization_noise,
        )
        return ValueIterationNetwork.softmax(state_action_values, self.beta)

    @staticmethod
    def value_iteration(
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        gamma: float = 0.8,
        iterations: int = 10,
        initialization_noise: float = 0.01,  # in standard deviation
        return_interim_estimates: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        state_action_values = np.zeros((n_states, n_actions))  # q(s, a)
        q_0, q_1 = (
            [np.empty((n_states // 2, n_actions))] * n_actions,
            [np.empty((n_states // 2, n_actions))] * n_actions,
        )

        if return_interim_estimates:
            state_action_values = np.zeros((iterations, n_states, n_actions))
            value_function = np.zeros((iterations, n_states))

        # dynamic programing algorithm
        for ii in tnrange(0, iterations):

            # calculate tiled values (we could randomize the tiling order but it isn't important)
            for a, (r_a, t_a) in enumerate(zip(reward_functions, transition_functions)):
                q_0[a] = r_a[tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_1)
            value_0 = np.max(q_0, axis=0)

            for a, (r_a, t_a) in enumerate(zip(reward_functions, transition_functions)):
                q_1[a] = r_a[~tiling] + gamma * t_a[~tiling][:, tiling].dot(value_0)
            value_1 = np.max(q_1, axis=0)

            if return_interim_estimates:
                value_function[ii, tiling] = value_0
                value_function[ii, ~tiling] = value_1
                state_action_values[ii, tiling, :] = np.array(q_0).T
                state_action_values[ii, ~tiling, :] = np.array(q_1).T

        if return_interim_estimates:
            return state_action_values, value_function

        value_function[tiling] = value_0
        value_function[~tiling] = value_1

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


class GridWorldNode(Node):
    def __init__(
        self,
        current_state: int,
        end_states: List[int],
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        state_reward_function: np.ndarray,
        n_actions: int = 4,
    ):
        self.current_state = current_state
        self.end_states = end_states
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.n_actions = n_actions
        self.hash = np.random.randint(6)

    def is_terminal(self) -> bool:
        return self.current_state in self.end_states

    def reward(self) -> float:
        return self.state_reward_function[self.current_state]

    def find_random_child(self) -> Node:
        # choose a random action
        a = choice(np.arange(0, self.n_actions))
        return self.take_action(a)

    def find_children(self):
        pass

    def take_action(self, action) -> Node:
        # sample a sucessor state
        t_a = self.transition_functions[action][self.current_state, :]
        sucessor_node = inverse_cmf_sampler(t_a)

        return GridWorldNode(
            current_state=sucessor_node,
            end_states=self.end_states,
            transition_functions=self.transition_functions,
            state_reward_function=self.state_reward_function,
            n_actions=self.n_actions,
        )

    def __hash__(self):
        "Nodes must be hashable"
        return self.hash

    def __eq__(self, other):
        "Nodes must be comparable"
        return self.hash == other.hash


class MCTSstub:
    @staticmethod
    def simulate(node):
        reward = 0
        while True:
            reward += node.reward()
            if node.is_terminal():
                return reward
            node = node.find_random_child()



