from random import choice
from typing import List, Union, Tuple

import numpy as np
from scipy import sparse

from simulation_utils import inverse_cmf_sampler


class GridWorldNode:
    def __init__(
        self,
        state: int,
        end_states: List[int],
        exploration_weight: float = 2 ** 0.5,
        n_actions: int = 4,
        epsilon: float = 0.01,  # minimum visitiation constant
    ):
        self.state = state
        self.end_states = end_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.exploration_weight = exploration_weight
        self.hash = int(self.state)

        self.q = None
        self.n = None
        self.children = None

    def is_terminal(self) -> bool:
        return self.state in self.end_states

    def is_unexplored(self) -> bool:
        if self.n is None:
            return True
        return False

    def find_random_child(self) -> int:
        # choose a random action
        return choice(np.arange(0, self.n_actions))

    def expand(self) -> int:
        self.children = self.find_children()
        self.q = np.zeros(len(self.children))
        self.n = np.zeros(len(self.children)) + self.epsilon
        return choice(list(self.children))

    def find_children(self):
        # children are the set of available actions for each node
        if self.is_terminal():
            return set()

        return {a0 for a0 in range(self.n_actions)}

    def __hash__(self):
        "Nodes must be hashable"
        return self.hash

    def __eq__(self, other):
        "Nodes must be comparable"
        return self.state == other.state

    def get_simulation_depth(self) -> int:
        if self.is_terminal():
            return 0

        node = self
        depth = 0
        while True:
            depth += 1
            if node.is_terminal():
                return depth
            action = node.find_random_child()
            node = node.draw_random_sucessor(action)

    def update_child_values(self, child_action: int, reward: float) -> None:
        if self.is_terminal():
            return
        self.q[child_action] += reward
        self.n[child_action] += 1

    def ucb_select_action(self) -> int:
        assert self.n.sum() >= 1, "Child Nodes have not been visited!"

        # deterministic UCB sampler
        r = self.q / self.n
        ucb = np.sqrt(np.log(self.n.sum()) / self.n)
        return np.argmax(r + self.exploration_weight * ucb)

    def get_argmax_policy(self):
        return int(np.argmax(self.q / self.n))


class MCTS:
    def __init__(
        self,
        end_states: List[int],
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        state_reward_function: np.ndarray,
        exploration_weight: float = 2 ** 0.5,
        n_actions: int = 4,
        epsilon: float = 0.01,  # minimum visitiation constant
    ):
        self.end_states = end_states
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.exploration_weight = exploration_weight
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.expanded_nodes = dict()

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
        return GridWorldNode(
            sucessor_state,
            self.end_states,
            exploration_weight=self.exploration_weight,
            n_actions=self.n_actions,
            epsilon=self.epsilon,
        )

    def _is_unexplored(self, node: GridWorldNode) -> bool:
        return node.state not in self.expanded_nodes

    def _draw_random_sucessor(self, node: GridWorldNode, action: int):
        # sample a sucessor state
        t_a = self.transition_functions[action][node.state, :]
        sucessor_state = inverse_cmf_sampler(t_a)

        return GridWorldNode(
            state=sucessor_state, end_states=self.end_states, n_actions=self.n_actions,
        )

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

    def select(
        self, node: GridWorldNode,
    ) -> Tuple[List[Tuple[GridWorldNode, int]], GridWorldNode]:

        # path is list of (state, action) tuples
        path = []

        while True:
            if self._is_unexplored(node) or node.is_terminal():
                return path, node

            # draw an action, update the path
            action = node.ucb_select_action()
            path.append((node, action))

            # draw next state
            node = self._sample_sucessor_node(node.state, action)

    def expand(self, node: GridWorldNode) -> int:
        if node.is_terminal():
            return -1
        self.expanded_nodes[node.state] = node

        self.q[node.state] = np.zeros(self.n_actions)
        self.n[node.state] = np.zeros(self.n_actions) + self.epsilon
        return node.expand()

    def simulate(self, node: GridWorldNode, action: int) -> float:
        if self._is_terminal(node):
            return self._get_reward(node)

        node = self._draw_random_sucessor(node, action)
        reward = 0

        while True:
            reward += self._get_reward(node)
            if node.is_terminal():
                return reward
            action = node.find_random_child()
            node = self._draw_random_sucessor(node, action)

    def backpropagate(self, path: List[Tuple[GridWorldNode, int]], reward: float):
        # path is a sequence of (state, action) tuples

        visited = set()  # only count nodes once?
        for (node, action) in reversed(path):
            if (node, action) not in visited:
                node.update_child_values(action, reward)

                #
                self._update_child_value(node, action, reward)

                visited.add((node, action))

    def do_rollout(self, start_state):
        "Make the tree one layer better. (Train for one iteration.)"
        path, leaf = self.select(start_state)
        a = self.expand(leaf)
        reward = self.simulate(leaf, a)
        path.append((leaf, a))
        self.backpropagate(path, reward)

    def get_policy(self):
        pi = np.ones_like(self.state_reward_function, dtype=int) * -1
        for state, node in self.expanded_nodes.items():
            pi[state] = node.get_argmax_policy()
        return pi
