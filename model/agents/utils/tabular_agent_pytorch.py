import torch
from torch import Tensor


def value_iteration(
    transition_function: Tensor,
    reward_function: Tensor,
    gamma: float,
    iterations: int = 100,
) -> Tensor:
    """Perform value iteration on a Markov Decision Process.

    Args:
        transition_function: Tensor of shape (n_states, n_actions, n_states)
        reward_function: Tensor of shape (n_states, n_actions)
        gamma: Discount factor

    Returns:
        torch.Tensor: Optimal value function of shape (n_states)
    """
    n_states, n_actions, _ = transition_function.shape
    q_values = torch.zeros(n_states, n_actions, device=transition_function.device)

    for _ in range(iterations):
        value_function = q_values.max(dim=1).values

        assert value_function.shape == (n_states,)
        assert reward_function.shape == (n_states, n_actions)
        assert transition_function.shape == (n_states, n_actions, n_states)
        q_values = reward_function + gamma * torch.matmul(transition_function, value_function)

    value_function = q_values.max(dim=1).values

    return q_values, value_function


class ModelBasedAgent:

    def __init__(
        self,
        n_states,
        n_actions,
        gamma=0.95,
        default_reward=0.0,
        device=None,
        iterations: int = 500,
        terminal_state: int | None = None,
    ):
        n_states = n_states + 1
        self.transitions = TransitionModel(n_states, n_actions, device, terminal_state=terminal_state)
        self.rewards = RewardModel(n_states, n_actions, device, terminal_state=terminal_state)

        self.n_states = n_states
        self.terminal_state = terminal_state
        self.gamma = gamma
        self.iterations = iterations

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        self.transitions.update(state, action, next_state, done)
        self.rewards.update(state, action, reward)

    def estimate_value_function(self, return_terminal_state: bool = False):
        transition_function = self.transitions.get_transition_function()
        reward_function = self.rewards.get_reward_function()
        _, value_function = value_iteration(
            transition_function,
            reward_function,
            gamma=self.gamma,
            iterations=self.iterations,
        )
        if return_terminal_state == False:
            mask = torch.ones(self.n_states, dtype=bool, device=value_function.device)
            mask[self.terminal_state] = False
            transition_function = value_function[mask]

        return value_function

    def get_graph_laplacian(self, normalized: bool = True, return_terminal_state: bool = False):
        return self.transitions.estimate_graph_laplacian(normalized, return_terminal_state)


class TransitionModel:
    def __init__(self, n_states, n_actions, device: str | None = None, terminal_state: int | None = None):
        device = device or torch.device("cpu")

        self.transition_counts = torch.zeros((n_states, n_actions, n_states), device=device)

        self.n_states = n_states
        self.n_actions = n_actions

        self.terminal_state = terminal_state if terminal_state else self.n_states - 1

        # initialize all the counts to go to the terminal state
        self.transition_counts[:, :, self.terminal_state] = 1e-6
        self.state_action_visited = torch.zeros((n_states, n_actions), device=device, dtype=torch.bool)

        # set the terminal state transitions to the terminal state
        self.transition_counts[self.terminal_state, :, self.terminal_state] = 1

    def update(self, state, action, next_state, done):
        if not self.state_action_visited[state, action]:
            self.transition_counts[state, action, :] = 0
            self.state_action_visited[state, action] = True

        self.transition_counts[state, action, next_state] += 1
        if done:
            for a in range(self.n_actions):
                self.transition_counts[state, a, self.terminal_state] += 1

    def get_transition_function(self):
        """Returns a tensor of shape (n_states, n_actions, n_states)"""
        return self.transition_counts / self.transition_counts.sum(dim=-1, keepdim=True)

    def estimate_graph_laplacian(self, normalized: bool = True, return_terminal_state: bool = False):
        """Estimate the graph Laplacian matrix of the transition model"""

        # Remove the terminal state from the transition function
        transition_function = self.get_transition_function()
        if return_terminal_state == False:
            mask = torch.ones(self.n_states, dtype=bool, device=self.transition_counts.device)
            mask[self.terminal_state] = False
            transition_function = self.get_transition_function()[mask][:, :, mask]

        adjacency_matrix = torch.tensor(transition_function.sum(dim=1) > 0, dtype=torch.float32)
        degree_matrix = adjacency_matrix.sum(dim=1).diag()

        if normalized:
            laplacian_matrix = torch.matmul(
                (degree_matrix**0.5), torch.matmul(adjacency_matrix, (degree_matrix**0.5))
            )
        else:
            laplacian_matrix = degree_matrix - adjacency_matrix

        return laplacian_matrix


class RewardModel:
    def __init__(self, n_states, n_actions, device: str | None = None, terminal_state: int | None = None):
        device = device or torch.device("cpu")

        self.reward_counts = torch.zeros((n_states, n_actions), device=device)
        self.reward_sums = torch.zeros((n_states, n_actions), device=device)

        self.n_actions = n_actions
        self.n_states = n_states

        self.terminal_state = terminal_state if terminal_state else self.n_states - 1

        # set the terminal states rewards to 0
        self.reward_counts[self.terminal_state, :] = 1
        self.reward_sums[self.terminal_state, :] = 0

    def update(self, state, action, reward):
        self.reward_counts[state, action] += 1
        self.reward_sums[state, action] += reward

    def get_reward_function(self):
        """Returns a tensor of shape (n_states, n_actions)"""
        return torch.nan_to_num(self.reward_sums / self.reward_counts, nan=0.0)
