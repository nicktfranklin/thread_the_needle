from typing import Optional, List

import numpy as np
from scipy.special import logsumexp

from models.value_iteration_network import ValueIterationNetwork
from models.utils import (
    inverse_cmf_sampler,
    one_hot,
    get_state_action_reward_from_sucessor_rewards,
    calculate_sr_from_transitions,
)


def get_optimal_sr_from_transitions(
    transition_functions, optimal_policy, gamma=0.95,
):

    # turn optimal policy in to pmf
    pi = optimal_policy * np.tile(1 / optimal_policy.sum(axis=1), (4, 1)).T

    # marginalize out the optimal policy
    T_sas = np.transpose(transition_functions, (1, 0, 2))
    T_ss = np.array([T_sas[ii, :].T.dot(pi[ii]) for ii in range(pi.shape[0])])

    # calculate the SR
    return calculate_sr_from_transitions(T_ss, gamma)


class SR:
    """
    This version of the SR sampling mechanism comes from Stachenfeld, 2017.  Here,
    we assume randomly sampled transitions from the True generative process.

    This needs to be framed as a simplified variant of the Russek model,
    or maybe even a new model we propose.
    """

    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        transition_functions: List[np.ndarray],
        state_reward_function: np.ndarray,
        beta: float = 1,
        n_rows: Optional[int] = None,
        n_columns: Optional[int] = None,
    ) -> None:

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.transition_functions = transition_functions
        self.reward_function = state_reward_function
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_states = self.transition_functions[0].shape[0]
        self.n_actions = len(self.transition_functions)
        self.beta = beta

        self.optimal_value_function: Optional[np.ndarray] = None

    def _calculate_optimal_value_function(
        self,
        iterations: int = 1000,
        n_rows: Optional[int] = None,
        n_columns: Optional[int] = None,
    ) -> np.ndarray:

        assert n_rows or self.n_rows, "n_rows not defined"
        assert n_columns or self.n_columns, "n_colmuns not defined"

        state_action_reward_functions = get_state_action_reward_from_sucessor_rewards(
            self.reward_function, self.transition_functions
        )

        _, state_value_function = ValueIterationNetwork.value_iteration(
            transition_functions=self.transition_functions,
            reward_functions=state_action_reward_functions,
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            gamma=self.gamma,
            iterations=iterations,
            return_interim_estimates=True,
        )

        return state_value_function

    def _calculate_optimal_sr(self) -> None:
        pass

    def _delta_update(
        self, initial_state: int, successor_state: int, M: np.ndarray
    ) -> np.ndarray:

        M[initial_state, :] = M[initial_state, :] + self.learning_rate * (
            one_hot(initial_state, self.n_states)
            + self.gamma * M[successor_state, :]
            - M[initial_state]
        )

        return M

    def _draw_random_successor_state(self, inital_state: int, action: int) -> int:
        return inverse_cmf_sampler(self.transition_functions[action][inital_state, :])

    def _sample_action_from_value_function(
        self, state: int, value_function: np.ndarray,
    ) -> int:

        # marginalize out the successor state for each action
        q_values = np.array(
            [
                self.gamma * self.transition_functions[a][state, :].dot(value_function)
                for a in range(self.n_actions)
            ]
        )

        log_pmf = self.beta * q_values - logsumexp(self.beta * q_values)
        return inverse_cmf_sampler(np.exp(log_pmf))

    def _one_sample_update(
        self, successor_representation: np.ndarray, value_function: np.ndarray
    ) -> np.ndarray:
        sampled_initial_state = np.random.randint(self.n_states)

        # sample according to precomputed value function
        sampled_action = self._sample_action_from_value_function(
            sampled_initial_state, value_function,
        )
        sampled_successor_state = self._draw_random_successor_state(
            sampled_initial_state, sampled_action
        )

        return self._delta_update(
            sampled_initial_state, sampled_successor_state, successor_representation
        )

    @staticmethod
    def get_value_function(
        successor_representation: np.ndarray, state_reward_function: np.ndarray
    ) -> np.ndarray:
        return successor_representation.dot(state_reward_function)
