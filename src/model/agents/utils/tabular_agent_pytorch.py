from typing import Optional, Tuple

import torch
from torch import Tensor


def value_iteration(
    transition_function: Tensor,
    reward_function: Tensor,
    gamma: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> Tuple[Tensor, Tensor]:
    """Perform value iteration on a Markov Decision Process.

    Args:
        transition_function: Tensor of shape (n_states, n_actions, n_states)
        reward_function: Tensor of shape (n_states, n_actions)
        gamma: Discount factor
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Tuple[Tensor, Tensor]: Tuple of (Q-values, optimal value function) with shapes
            ((n_states, n_actions), (n_states,))
    """
    n_states, n_actions, _ = transition_function.shape
    q_values = torch.zeros(n_states, n_actions, device=transition_function.device)

    for _ in range(max_iterations):
        old_q_values = q_values.clone()
        value_function = q_values.max(dim=1).values

        assert value_function.shape == (n_states,)
        assert reward_function.shape == (n_states, n_actions)
        assert transition_function.shape == (n_states, n_actions, n_states)

        q_values = reward_function + gamma * torch.matmul(
            transition_function, value_function
        )

        # Check convergence
        if torch.max(torch.abs(q_values - old_q_values)) < epsilon:
            break

    value_function = q_values.max(dim=1).values
    return q_values, value_function


class ModelBasedAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.95,
        device: Optional[str] = None,
        iterations: int = 500,
        terminal_state: Optional[int] = None,
        epsilon: float = 1e-6,
    ):
        """Initialize the model-based RL agent.

        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions in the MDP
            gamma: Discount factor
            device: Device to use for computations
            iterations: Maximum number of iterations for value iteration
            terminal_state: Terminal state index (if any)
            epsilon: Convergence threshold for value iteration
        """
        n_states = n_states + 1  # Add terminal state
        self.device = device or torch.device("cpu")

        self.transitions = TransitionModel(
            n_states, n_actions, self.device, terminal_state=terminal_state
        )
        self.rewards = RewardModel(
            n_states, n_actions, self.device, terminal_state=terminal_state
        )

        self.n_states = n_states
        self.n_actions = n_actions
        self.terminal_state = terminal_state
        self.gamma = gamma
        self.iterations = iterations
        self.epsilon = epsilon

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Update the agent's models with a new transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        if not (0 <= state < self.n_states and 0 <= action < self.n_actions):
            raise ValueError(f"Invalid state {state} or action {action}")

        self.transitions.update(state, action, next_state, done)
        self.rewards.update(state, action, reward)

    def estimate_value_function(self, return_terminal_state: bool = False) -> Tensor:
        """Estimate the optimal value function using value iteration.

        Args:
            return_terminal_state: Whether to include the terminal state in the output

        Returns:
            Tensor: Optimal value function
        """
        transition_function = self.transitions.get_transition_function()
        reward_function = self.rewards.get_reward_function()

        _, value_function = value_iteration(
            transition_function,
            reward_function,
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_iterations=self.iterations,
        )

        if not return_terminal_state:
            mask = torch.ones(self.n_states, dtype=bool, device=self.device)
            mask[self.terminal_state] = False
            value_function = value_function[mask]

        return value_function

    def get_policy(self, deterministic: bool = True) -> Tensor:
        """Get the current policy of the agent.

        Args:
            deterministic: Whether to return a deterministic policy

        Returns:
            Tensor: Policy matrix of shape (n_states, n_actions) containing probabilities
                   or a one-hot vector for deterministic policies
        """
        transition_function = self.transitions.get_transition_function()
        reward_function = self.rewards.get_reward_function()

        q_values, _ = value_iteration(
            transition_function,
            reward_function,
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_iterations=self.iterations,
        )

        if deterministic:
            return torch.eye(self.n_actions, device=self.device)[q_values.argmax(dim=1)]
        else:
            # Boltzmann policy
            temperature = 1.0
            policy = torch.softmax(q_values / temperature, dim=1)
            return policy

    def get_graph_laplacian(
        self, normalized: bool = True, return_terminal_state: bool = False
    ) -> Tensor:
        """Get the graph Laplacian of the transition model.

        Args:
            normalized: Whether to return the normalized Laplacian
            return_terminal_state: Whether to include the terminal state

        Returns:
            Tensor: Graph Laplacian matrix
        """
        return self.transitions.estimate_graph_laplacian(
            normalized, return_terminal_state
        )


class TransitionModel:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        device: Optional[str] = None,
        terminal_state: Optional[int] = None,
    ):
        self.device = device or torch.device("cpu")
        self.transition_counts = torch.zeros(
            (n_states, n_actions, n_states), device=self.device
        )

        self.n_states = n_states
        self.n_actions = n_actions
        self.terminal_state = terminal_state if terminal_state else self.n_states - 1

        # Initialize counts to go to terminal state
        self.transition_counts[:, :, self.terminal_state] = 1e-6
        self.state_action_visited = torch.zeros(
            (n_states, n_actions), device=self.device, dtype=torch.bool
        )

        # Set terminal state transitions
        self.transition_counts[self.terminal_state, :, self.terminal_state] = 1

    def update(self, state: int, action: int, next_state: int, done: bool) -> None:
        """Update the transition model with a new observation."""
        if not (0 <= state < self.n_states and 0 <= action < self.n_actions):
            raise ValueError(f"Invalid state {state} or action {action}")

        if not self.state_action_visited[state, action]:
            self.transition_counts[state, action, :] = 0
            self.state_action_visited[state, action] = True

        self.transition_counts[state, action, next_state] += 1
        if done:
            self.transition_counts[state, action, self.terminal_state] += 1

    def get_transition_function(self) -> Tensor:
        """Get the current transition function probabilities."""
        return self.transition_counts / (
            self.transition_counts.sum(dim=-1, keepdim=True) + 1e-10
        )

    def estimate_graph_laplacian(
        self, normalized: bool = True, return_terminal_state: bool = False
    ) -> Tensor:
        """Estimate the graph Laplacian matrix of the transition model."""
        transition_function = self.get_transition_function()

        if not return_terminal_state:
            mask = torch.ones(self.n_states, dtype=bool, device=self.device)
            mask[self.terminal_state] = False
            transition_function = transition_function[mask][:, :, mask]

        adjacency_matrix = (transition_function.sum(dim=1) > 0).float()
        degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))

        if normalized:
            # Using pseudo-inverse for numerical stability
            sqrt_degree = torch.diag(
                1.0 / torch.sqrt(torch.clamp(degree_matrix.diag(), min=1e-10))
            )
            laplacian_matrix = torch.matmul(
                sqrt_degree, torch.matmul(adjacency_matrix, sqrt_degree)
            )
        else:
            laplacian_matrix = degree_matrix - adjacency_matrix

        return laplacian_matrix


class RewardModel:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        device: Optional[str] = None,
        terminal_state: Optional[int] = None,
    ):
        self.device = device or torch.device("cpu")
        self.reward_counts = torch.zeros((n_states, n_actions), device=self.device)
        self.reward_sums = torch.zeros((n_states, n_actions), device=self.device)

        self.n_states = n_states
        self.n_actions = n_actions
        self.terminal_state = terminal_state if terminal_state else self.n_states - 1

        # Set terminal state rewards to 0
        self.reward_counts[self.terminal_state, :] = 1
        self.reward_sums[self.terminal_state, :] = 0

    def update(self, state: int, action: int, reward: float) -> None:
        """Update the reward model with a new observation."""
        if not (0 <= state < self.n_states and 0 <= action < self.n_actions):
            raise ValueError(f"Invalid state {state} or action {action}")

        self.reward_counts[state, action] += 1
        self.reward_sums[state, action] += reward

    def get_reward_function(self) -> Tensor:
        """Get the current reward function estimates."""
        return torch.nan_to_num(
            self.reward_sums / (self.reward_counts + 1e-10), nan=0.0
        )
