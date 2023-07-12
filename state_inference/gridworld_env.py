from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, SupportsFloat, TypeVar, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import fftconvolve

from state_inference.utils.pytorch_utils import make_tensor
from state_inference.utils.utils import one_hot
from value_iteration.environments.thread_the_needle import (
    GridWorld,
    make_thread_the_needle_walls,
)

ObsType = TypeVar("ObsType", np.ndarray, torch.Tensor)
ActType = TypeVar("ActType")
StateType = TypeVar("StateType")
RewType = TypeVar("RewType")


# The state-location is smoothed by convoling an RBF kernel with the
# one-hot representation of the location in the nXn Grid
class RbfKernelEmbedding:
    def __init__(self, kernel_size: int = 51, len_scale: float = 0.15) -> None:
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.kernel = torch.empty([kernel_size, kernel_size]).float()
        center = torch.ones([1, 1]) * kernel_size // 2

        def _rbf(x: torch.Tensor) -> float:
            dist = ((x - center) ** 2).sum() ** 0.5
            return torch.exp(-len_scale * dist)

        for r in range(kernel_size):
            for c in range(kernel_size):
                self.kernel[r, c] = _rbf(torch.Tensor([r, c]))

    def __call__(self, input_space: torch.Tensor) -> torch.Tensor:
        assert len(input_space.shape) == 2
        return fftconvolve(input_space, self.kernel, mode="same")


class ObservationModel:
    def __init__(
        self,
        h: int = 10,
        w: int = 10,
        map_height: int = 60,
        rbf_kernel_size: int = 51,
        rbf_kernel_scale: float = 0.15,
        location_noise_scale=1.0,
        noise_log_mean: float = -2,
        noise_log_scale: float = 0.05,
        noise_corruption_prob=0.01,
    ) -> None:
        assert map_height % h == 0
        assert map_height % w == 0
        multiplier = map_height // h
        self.h = h
        self.w = w
        self.map_height = map_height

        x = [ii for ii in range(h) for _ in range(h)]
        y = [ii for _ in range(w) for ii in range(w)]
        self.coordinates = np.array([x, y]).T * multiplier + multiplier // 2
        self.states = {ii: c for ii, c in enumerate(self.coordinates)}

        self.kernel = RbfKernelEmbedding(rbf_kernel_size, rbf_kernel_scale)

        self.loc_noise_scale = location_noise_scale
        self.noise_log_mean = noise_log_mean
        self.noise_log_scale = noise_log_scale
        self.noise_corruption_prob = noise_corruption_prob

    def get_obs_coords(self, s: StateType) -> tuple[int, int]:
        return self.states[s]

    def get_grid_coords(self, s: StateType) -> tuple[int, int]:
        r = s // self.w
        c = s % self.w
        return r, c

    def get_grid_location(self, s: StateType) -> ObsType:
        r, c = self.get_grid_coords(s)
        raw_state = np.zeros((self.h, self.w))
        raw_state[r, c] = 1
        return raw_state

    def _embed_one_hot(self, x: int, y: int) -> torch.Tensor:
        grid = torch.zeros((self.map_height, self.map_height))
        grid[x, y] = 1.0
        return grid

    @make_tensor
    def embed_state(self, s: StateType) -> torch.Tensor:
        x, y = self.get_obs_coords(s)
        grid = self._embed_one_hot(x, y)
        return self.kernel(grid)

    def _location_corruption(self, x: int, y: int) -> tuple[int, int]:
        x += int(round(np.random.normal(loc=0, scale=self.loc_noise_scale)))
        y += int(round(np.random.normal(loc=0, scale=self.loc_noise_scale)))

        # truncate
        if x < 0:
            x = 0
        if x >= self.map_height:
            x = self.map_height - 1
        if y < 0:
            y = 0
        if y >= self.map_height:
            y = self.map_height - 1

        return x, y

    def _random_embedding_noise(self) -> torch.Tensor:
        corrupted_mask = torch.exp(
            torch.randn(self.map_height, self.map_height) * self.noise_log_scale
            + self.noise_log_mean
        )
        corrupted_mask *= (
            torch.rand(self.map_height, self.map_height) < self.noise_corruption_prob
        )
        return corrupted_mask

    @make_tensor
    def embed_state_corrupted(self, s: StateType) -> torch.Tensor:
        x, y = self.get_obs_coords(s)
        x, y = self._location_corruption(x, y)
        grid = self._embed_one_hot(x, y)
        grid += self._random_embedding_noise()
        return self.kernel(grid)

    def __call__(
        self, s: Union[int, List[int]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(s, list):
            return [self.embed_state_corrupted(s0) for s0 in s]
        return self.embed_state_corrupted(s)

    def display_state(self, s: StateType) -> None:
        # x, y = self.get_obs_coords(s)
        raw_state = self.get_grid_location(s)
        _, axes = plt.subplots(1, 3, figsize=(10, 20))
        axes[0].imshow(1 - raw_state, cmap="gray")

        def plt_lines(h, w, ax):
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()
            for h0 in range(h):
                ax.plot([h0 + 0.5, h0 + 0.5], [-0.5, w + 0.5], c="gray")
            for w0 in range(w):
                ax.plot([-0.5, h + 0.5], [w0 + 0.5, w0 + 0.5], c="gray")
                ax.set_ylim(y_lim)
                ax.set_xlim(x_lim)

        plt_lines(self.h, self.w, axes[0])
        axes[1].imshow(self.embed_state(s))
        axes[2].imshow(self.embed_state_corrupted(s))
        axes[0].set_title("Grid-States")
        axes[1].set_title("Noise-free Observations")
        axes[2].set_title("Noisy Observation")
        plt.show()


class TransitionModel:
    # Generative model. Assumes connectivity between neighboring states in a 2d gridworld

    def __init__(
        self,
        h: int,
        w: int,
        walls: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self.random_transitions = self._make_random_transitions(h, w)
        self.state_action_transitions = self._make_cardinal_transition_function(
            h, w, walls
        )
        self.adjecency_list = self._make_adjecency_list(self.random_transitions)
        self.walls = walls
        self.n_states = h * w
        self.h = h
        self.w = w

    @staticmethod
    def _make_cardinal_transition_function(
        h: int,
        w: int,
        walls: Optional[list[tuple[int, int]]] = None,
    ) -> Dict[str, np.ndarray]:
        states = np.arange(h * w).reshape(h, w)
        transitions = {
            "up": np.vstack([states[0, :], states[:-1, :]]),
            "down": np.vstack([states[1:, :], states[-1, :]]),
            "left": np.hstack([states[:, :1], states[:, :-1]]),
            "right": np.hstack([states[:, 1:], states[:, -1:]]),
        }
        transitions = {k: one_hot(v.reshape(-1), h * w) for k, v in transitions.items()}

        if walls is not None:

            def filter_transition_function(t):
                def _filter_wall(b1, b2):
                    t[b1][b1], t[b1][b2] = t[b1][b1] + t[b1][b2], 0
                    t[b2][b2], t[b2][b1] = t[b2][b2] + t[b2][b1], 0

                for b1, b2 in walls:
                    _filter_wall(b1, b2)
                return t

            transitions = {
                k: filter_transition_function(t) for k, t in transitions.items()
            }

        return transitions

    @staticmethod
    def _make_random_transitions(
        h: int, w: int, walls: Optional[list[tuple[StateType, StateType]]] = None
    ) -> np.ndarray:
        t = np.zeros((h * w, h * w))
        for s0 in range(h * w):
            # down
            if s0 - w >= 0:
                t[s0, s0 - w] = 1
            # up
            if s0 + w + 1 < h * w:
                t[s0, s0 + w] = 1

            # right
            if (s0 + 1) % w > 0:
                t[s0, s0 + 1] = 1

            # left
            if s0 % w > 0:
                t[s0, s0 - 1] = 1

        # normalize
        t = TransitionModel._normalize(t)
        if walls:
            return TransitionModel._add_walls(t, walls)
        return t

    @staticmethod
    def _add_walls(
        transitions: dict[StateType, StateType],
        walls: list[tuple[StateType, StateType]],
    ) -> dict[StateType, StateType]:
        for s, sp in walls:
            transitions[s, sp] = 0
            transitions[sp, s] = 0
        return TransitionModel._normalize(transitions)

    @staticmethod
    def _normalize(t: np.ndarray) -> np.ndarray:
        return t / np.tile(t.sum(axis=1).reshape(-1, 1), t.shape[1])

    @staticmethod
    def _make_adjecency_list(transitions: np.ndarray) -> Dict[int, np.ndarray]:
        edges = {}
        for s, t in enumerate(transitions):
            edges[s] = np.where(t > 0)[0]
        return edges

    def get_sucessor_distribution(
        self, state: StateType, action: ActType
    ) -> np.ndarray:
        assert state in self.adjecency_list
        assert action in self.state_action_transitions.keys()
        return self.state_action_transitions[action][state]

    def display_gridworld(
        self, ax: Optional[matplotlib.axes.Axes] = None, wall_color="k"
    ) -> matplotlib.axes.Axes:
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))
            ax.invert_yaxis()

        ax.set_yticks([])
        ax.set_xticks([])
        # plot the gridworld tiles
        for r in range(self.h):
            ax.plot([-0.5, self.w - 0.5], [r - 0.5, r - 0.5], c="grey", lw=0.5)
        for c in range(self.w):
            ax.plot([c - 0.5, c - 0.5], [-0.5, self.h - 0.5], c="grey", lw=0.5)

        for s0, s1 in self.walls:
            r0, c0 = GridWorld.get_position_from_state(s0, self.w)
            r1, c1 = GridWorld.get_position_from_state(s1, self.w)

            x = (r0 + r1) / 2
            y = (c0 + c1) / 2

            assert (r0 == r1) or (c0 == c1), f"Not a valid wall! {r0} {r1} {c0} {s1}"
            if c0 == c1:
                ax.plot([y - 0.5, y + 0.5], [x, x], c=wall_color, lw=3)
            else:
                ax.plot([y, y], [x - 0.5, x + 0.5], c=wall_color, lw=3)

        return ax


class RewardModel:
    def __init__(
        self,
        state_rewards: Optional[Dict[StateType, RewType]] = None,
        state_action_rewards: Optional[Dict[tuple[StateType, ActType], RewType]] = None,
    ) -> None:
        # only one of the reward functions should be specified.
        assert (state_rewards is not None) ^ (state_action_rewards is not None)

        self.state_rewards = state_rewards
        self.state_action_rewards = state_action_rewards

    def get_reward(self, state: StateType, action: Optional[ActType] = None):
        if self.state_rewards is not None:
            return self.state_rewards.get(state, 0)

        return self.state_action_rewards.get((state, action), 0)


class Env(ABC):
    """Modeled after the gymnasium API"""

    @abstractmethod
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        ...

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        ...


class GridWorldEnv(Env):
    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        observation_model: ObservationModel,
        initial_state: Optional[int] = None,
        n_states: Optional[int] = None,
        end_state: Optional[list[int]] = None,
        random_initial_state: bool = False,
    ) -> None:
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.observation_model = observation_model

        # self.n_states = n_states if n_states else self.transition_model.n_states
        self.n_states = n_states
        self.initial_state = initial_state
        assert (
            random_initial_state is not None or initial_state is not None
        ), "must specify either the inital location or random initialization"
        self.current_state = self._get_initial_state()

        self.observation = self._generate_observation(self.current_state)
        self.states = np.arange(n_states)
        self.end_state = end_state

    def _check_terminate(self, state: int) -> bool:
        if self.end_state == None:
            return False
        return state in self.end_state

    def _get_initial_state(self) -> int:
        if self.initial_state:
            return self.initial_state
        return np.random.randint(self.n_states)

    def _generate_observation(self, state: int) -> torch.Tensor:
        return self.observation_model(state)

    def reset(self) -> torch.Tensor:
        self.current_state = self._get_initial_state()
        return self._generate_observation(self.current_state)

    def get_obseservation(self) -> torch.Tensor:
        return self.observation

    def get_state(self) -> torch.Tensor:
        return self.current_state

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        pdf_s = self.transition_model.get_sucessor_distribution(
            self.current_state, action
        )

        assert np.sum(pdf_s) == 1, (action, self.current_state, pdf_s)
        assert np.all(pdf_s >= 0), print(pdf_s)

        sucessor_state = np.random.choice(self.states, p=pdf_s)
        sucessor_observation = self._generate_observation(sucessor_state)

        reward = self.reward_model.get_reward(sucessor_state)
        terminated = self._check_terminate(sucessor_state)
        truncated = False
        info = dict()

        output = tuple([sucessor_observation, reward, terminated, truncated, info])

        self.current_state = sucessor_state
        self.observation = sucessor_observation

        return output


class ThreadTheNeedleEnv(GridWorldEnv):
    @classmethod
    def create_env(
        cls,
        h: int,
        w: int,
        map_height: int,
        rewards: dict[StateType, RewType],
        observation_kwargs: dict[str, Any],
        **gridworld_env_kwargs,
    ):
        # Define the transitions for the thread the needle task
        walls = make_thread_the_needle_walls(20)
        transition_model = TransitionModel(h, w, walls)

        observation_model = ObservationModel(h, w, map_height, **observation_kwargs)

        reward_model = RewardModel(rewards)

        return cls(
            transition_model, reward_model, observation_model, **gridworld_env_kwargs
        )
