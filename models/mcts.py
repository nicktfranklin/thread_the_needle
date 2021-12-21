from abc import ABC, abstractmethod
from collections import namedtuple
from random import choice
from typing import List, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm

from models.value_iteration_network import softmax
from simulation_utils import inverse_cmf_sampler

N_ACTIONS = 4


class _Node(ABC):
    @abstractmethod
    def find_random_child(self) -> int:
        return 0

    @abstractmethod
    def expand(self) -> int:
        return 0

    @abstractmethod
    def __hash__(self):
        return 123456789

    @abstractmethod
    def __eq__(self, other):
        return True


_GWN = namedtuple("GridWorldTuple", "state n_actions")


class GridWorldNode(ABC, _GWN):
    def find_random_child(self) -> int:
        # choose a random action
        return choice(np.arange(0, self.n_actions))

    def expand(self) -> int:
        return self.find_random_child()


class MCTS:
    def __init__(
        self,
        end_states: List[int],
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        state_reward_function: np.ndarray,
        exploration_weight: float = 2 ** 0.5,
        n_actions: int = 4,
        epsilon: float = 1,  # minimum visitiation constant
        max_depth: int = 1000,
        n_sims: int = 1,
        gamma: float = 0.99
    ):
        self.end_states = end_states
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.exploration_weight = exploration_weight
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.expanded_nodes = dict()
        self.n_sims = n_sims
        self.gamma = gamma

        self.q = dict()
        self.n = dict()

    def _sample_sucessor_state(self, state: int, action: int) -> int:
        t_a = self.transition_functions[action][state, :]
        sucessor_state = inverse_cmf_sampler(t_a)
        return sucessor_state

    def _sample_sucessor_node(self, state: int, action: int) -> GridWorldNode:
        sucessor_state = self._sample_sucessor_state(state, action)
        if sucessor_state in self.expanded_nodes:
            return self.expanded_nodes[sucessor_state]
        return GridWorldNode(state=sucessor_state, n_actions=self.n_actions)

    def _is_unexplored(self, node: GridWorldNode) -> bool:
        return node.state not in self.expanded_nodes

    def _draw_random_sucessor(self, node: GridWorldNode, action: int):
        # sample a sucessor state
        t_a = self.transition_functions[action][node.state, :]
        sucessor_state = inverse_cmf_sampler(t_a)

        return GridWorldNode(state=sucessor_state, n_actions=self.n_actions)

    def _get_reward(self, node: GridWorldNode) -> bool:
        return self.state_reward_function[node.state]

    def _is_expanded(self, node: GridWorldNode) -> bool:
        return node in self.q.keys()

    def _is_terminal(self, node: GridWorldNode) -> bool:
        return node.state in self.end_states

    def _update_child_value(
        self, node: GridWorldNode, child_action: int, reward: float
    ) -> None:
        if self._is_terminal(node):
            return

        self.q[node.state][child_action] += reward
        self.n[node.state][child_action] += 1

    def _get_ucb_values(self, node: GridWorldNode) -> np.ndarray:
        r = self.q[node.state] / self.n[node.state]
        ucb = np.sqrt(2*np.log(self.n[node.state].sum()) / self.n[node.state])
        return r + self.exploration_weight * ucb

    def _ucb_select_action(self, node: GridWorldNode) -> int:
        assert self.n[node.state].sum() >= 1, "Child Nodes have not been visited!"

        # deterministic UCB sampler
        return np.argmax(self._get_ucb_values(node))

    def _get_argmax_policy(self, node: GridWorldNode) -> int:
        return int(np.argmax(self.q[node.state] / self.n[node.state]))

    def select(
        self, node: GridWorldNode,
    ) -> Tuple[List[Tuple[GridWorldNode, int]], GridWorldNode]:

        # path is list of (state, action) tuples
        path = []
        reward, depth = 0, 0

        while depth < self.max_depth:
            if self._is_unexplored(node) or self._is_terminal(node):
                return path, node
            # self.q[node.state] += self._get_reward(node)

            # draw an action, update the path
            action = self._ucb_select_action(node)
            path.append((node, action))

            # draw next state
            node = self._sample_sucessor_node(node.state, action)
            reward += self._get_reward(node)

        return path, node

    def expand(self, node: GridWorldNode) -> int:
        if self._is_terminal(node):
            return -1
        self.expanded_nodes[node.state] = node

        self.q[node.state] = np.zeros(self.n_actions)
        self.n[node.state] = np.zeros(self.n_actions) + self.epsilon
        return node.expand()

    def _single_sim(self, node: GridWorldNode, action: int) -> float:

        if self._is_terminal(node):
            return self._get_reward(node)

        node = self._draw_random_sucessor(node, action)
        reward = 0
        depth = 0
        while True:
            reward += self._get_reward(node) * (self.gamma ** depth)
            if self._is_terminal(node):
                return reward
            action = node.find_random_child()
            node = self._draw_random_sucessor(node, action)
            depth += 1

    def simulate(self, node: GridWorldNode, action: int, k=None) -> float:
        iterations = k or self.n_sims
        return np.mean([self._single_sim(node, action) for _ in range(iterations)])

    def backpropagate(self, path: List[Tuple[GridWorldNode, int]], reward: float):
        # path is a sequence of (state, action) tuples
        for (node, action) in reversed(path):
            self._update_child_value(node, action, reward)

    def do_single_rollout(self, start_state):
        "Make the tree one layer better. (Train for one iteration.)"
        path, leaf = self.select(start_state)
        a = self.expand(leaf)
        reward = self.simulate(leaf, a)
        path.append((leaf, a))
        self.backpropagate(path, reward)

    def do_rollouts(self, start_state, k=10):
        for _ in tqdm(range(k), desc="MCTS Rollouts"):
            self.do_single_rollout(start_state)

    def get_policy(self):
        pi = np.ones_like(self.state_reward_function, dtype=int) * -1
        for state, node in self.expanded_nodes.items():
            pi[state] = self._get_argmax_policy(node)
        return pi

    def get_simulation_depth(self, node: GridWorldNode) -> int:
        if self._is_terminal(node):
            return 0

        depth = 0
        while True:
            depth += 1
            if self._is_terminal(node):
                return depth
            action = node.find_random_child()
            node = self._draw_random_sucessor(node, action)

    def get_selection_policy(self, beta: float) -> np.ndarray:
        policy = (
            np.ones((self.state_reward_function.shape[0], N_ACTIONS), dtype=float)
            / N_ACTIONS
        )
        for state, node in self.expanded_nodes.items():
            w = self._get_ucb_values(node)
            policy[state, :] = softmax(w, beta)
        return policy

