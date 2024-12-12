from typing import Tuple

import pytest
import torch

from src.model.agents.utils.tabular_agent_pytorch import (
    ModelBasedAgent,
    RewardModel,
    TransitionModel,
    value_iteration,
)


@pytest.fixture
def simple_mdp() -> Tuple[ModelBasedAgent, torch.Tensor, torch.Tensor]:
    """Create a simple 2-state MDP for testing."""
    device = torch.device("cpu")
    agent = ModelBasedAgent(n_states=2, n_actions=2, device=device)

    # True transition probabilities (excluding terminal state)
    true_transitions = torch.tensor(
        [
            [[0.7, 0.3], [0.2, 0.8]],  # state 0
            [[0.4, 0.6], [0.9, 0.1]],  # state 1
        ],
        device=device,
    )

    # True rewards
    true_rewards = torch.tensor(
        [
            [1.0, -1.0],  # state 0
            [-1.0, 1.0],  # state 1
        ],
        device=device,
    )

    return agent, true_transitions, true_rewards


def test_initialization():
    """Test agent initialization."""
    agent = ModelBasedAgent(n_states=3, n_actions=2)

    assert agent.n_states == 4  # 3 states + terminal
    assert agent.n_actions == 2
    assert agent.terminal_state == 3
    assert agent.gamma == 0.95

    # Check tensor shapes
    assert agent.q_values.shape == (4, 2)
    assert agent.value_function.shape == (4,)


def test_terminal_state_properties(simple_mdp):
    """Test terminal state initialization and properties."""
    agent, _, _ = simple_mdp

    # Check terminal state transitions
    trans_func = agent.transitions.get_transition_function()
    term_state = agent.terminal_state

    # Terminal state should be absorbing
    assert torch.allclose(
        trans_func[term_state, :, term_state], torch.ones(agent.n_actions)
    )

    # Terminal state should have zero rewards
    rewards = agent.rewards.get_reward_function()
    assert torch.allclose(rewards[term_state], torch.zeros(agent.n_actions))


def test_state_expansion():
    """Test dynamic state space expansion."""
    agent = ModelBasedAgent(n_states=2, n_actions=2)
    initial_states = agent.n_states

    # Add transition to new state
    agent.update(state=0, action=0, reward=1.0, next_state=agent.n_states, done=False)

    assert agent.n_states == initial_states + 1
    assert agent.rewards.reward_counts.shape[0] == agent.n_states
    assert agent.value_function.shape == (agent.n_states,)
    assert agent.q_values.shape == (agent.n_states, agent.n_actions)

    agent.update(state=agent.n_states, action=0, reward=1.0, next_state=0, done=False)

    assert agent.n_states == initial_states + 2
    assert agent.rewards.reward_counts.shape[0] == agent.n_states
    assert agent.value_function.shape == (agent.n_states,)
    assert agent.q_values.shape == (agent.n_states, agent.n_actions)


def test_update_and_estimation(simple_mdp):
    """Test model updates and value estimation."""
    agent, true_transitions, true_rewards = simple_mdp

    # Add some transitions
    agent.update(0, 0, 1.0, 1, False)
    agent.update(0, 0, 1.0, 1, False)
    agent.update(0, 0, 1.0, 0, False)
    agent.update(1, 1, -1.0, 0, True)

    # Test transition estimation
    trans_func = agent.transitions.get_transition_function()
    assert trans_func[0, 0, 1] > trans_func[0, 0, 0]  # More transitions to state 1

    # Test reward estimation
    reward_func = agent.rewards.get_reward_function()
    assert reward_func[0, 0] == 1.0
    assert reward_func[1, 1] == -1.0


def test_value_iteration():
    """Test value iteration convergence."""
    # Create a simple deterministic MDP
    trans_func = torch.zeros(2, 2, 2)
    trans_func[0, 0, 1] = 1.0  # action 0: state 0 -> state 1
    trans_func[0, 1, 0] = 1.0  # action 1: state 0 -> state 0
    trans_func[1, :, 1] = 1.0  # state 1 is absorbing

    reward_func = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    q_values, value_func = value_iteration(trans_func, reward_func, gamma=0.9)

    # Value of state 0 should be greater than state 1
    assert value_func[0] > value_func[1]
    # Optimal action in state 0 should be action 0
    assert q_values[0, 0] > q_values[0, 1]


def test_policy_generation(simple_mdp):
    """Test policy generation."""
    agent, _, _ = simple_mdp

    # Add some transitions
    agent.update(0, 0, 1.0, 1, False)
    agent.update(1, 1, -1.0, 0, False)

    # Test deterministic policy
    det_policy = agent.get_policy(deterministic=True)
    assert det_policy.shape == (agent.n_states, agent.n_actions)
    assert torch.all(torch.sum(det_policy, dim=1) == 1)

    # Test stochastic policy
    stoch_policy = agent.get_policy(deterministic=False)
    assert stoch_policy.shape == (agent.n_states, agent.n_actions)
    assert torch.allclose(torch.sum(stoch_policy, dim=1), torch.ones(agent.n_states))


def test_graph_laplacian(simple_mdp):
    """Test graph Laplacian computation."""
    agent, _, _ = simple_mdp

    # Add some transitions
    agent.update(0, 0, 1.0, 1, False)
    agent.update(1, 1, -1.0, 0, False)

    # Test normalized Laplacian
    norm_lap = agent.get_graph_laplacian(normalized=True)
    assert norm_lap.shape == (2, 2)  # excluding terminal state
    assert torch.allclose(norm_lap, norm_lap.T)  # should be symmetric

    # Test unnormalized Laplacian
    unnorm_lap = agent.get_graph_laplacian(normalized=False)
    assert unnorm_lap.shape == (2, 2)
    assert torch.allclose(unnorm_lap, unnorm_lap.T)


def test_error_handling():
    """Test error handling."""
    agent = ModelBasedAgent(n_states=2, n_actions=2)

    # Test invalid state/action
    with pytest.raises(ValueError):
        agent.update(-1, 0, 1.0, 0, False)

    with pytest.raises(ValueError):
        agent.update(0, -1, 1.0, 0, False)

    # Test invalid gamma in value iteration
    trans_func = torch.zeros(2, 2, 2)
    reward_func = torch.zeros(2, 2)

    with pytest.raises(AssertionError):
        value_iteration(trans_func, reward_func, gamma=1.1)
