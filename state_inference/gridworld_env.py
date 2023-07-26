from random import choices
from typing import Any, Dict, List, Optional, TypeVar, Union

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import fftconvolve

from state_inference.utils.pytorch_utils import make_tensor, normalize
from state_inference.utils.utils import one_hot
from value_iteration.environments.thread_the_needle import (
    GridWorld,
    make_thread_the_needle_walls,
)
from value_iteration.models.value_iteration_network import ValueIterationNetwork

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
StateType = TypeVar("StateType")
RewType = TypeVar("RewType")


# The state-location is smoothed by convoling an RBF kernel with the
# one-hot representation of the location in the nXn Grid
class RbfKernelEmbedding:
    def __init__(self, kernel_size: int = 51, len_scale: float = 0.15) -> None:
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.kernel = np.empty([kernel_size, kernel_size], dtype=float)
        center = np.ones([1, 1], dtype=float) * kernel_size // 2

        def _rbf(x: np.ndarray) -> float:
            dist = ((x - center) ** 2).sum() ** 0.5
            return np.exp(-len_scale * dist)

        for r in range(kernel_size):
            for c in range(kernel_size):
                self.kernel[r, c] = _rbf(np.array([r, c]))

    def __call__(self, input_space: np.ndarray) -> np.ndarray:
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
        noise_corruption_prob: float = 0.01,
        discrete_range: Optional[tuple[int, int]] = None,
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
        self.discrete_range = (
            discrete_range if discrete_range is not None else tuple([0, 255])
        )

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

    def _embed_one_hot(self, x: int, y: int) -> np.ndarray:
        grid = np.zeros((self.map_height, self.map_height))
        grid[x, y] = 1.0
        return grid

    def _make_embedding_from_grid(self, grid: np.ndarray) -> np.ndarray:
        embedding = self.kernel(grid)

        # convert from float to 8bit
        embedding = normalize(embedding)
        return (embedding * 255).astype(int)

    def embed_state(self, s: StateType) -> np.ndarray:
        x, y = self.get_obs_coords(s)
        grid = self._embed_one_hot(x, y)
        return self._make_embedding_from_grid(grid)

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

    def _random_embedding_noise(self) -> np.ndarray:
        corrupted_mask = np.exp(
            np.random.randn(self.map_height, self.map_height) * self.noise_log_scale
            + self.noise_log_mean
        )
        corrupted_mask *= (
            np.random.uniform(size=(self.map_height, self.map_height))
            < self.noise_corruption_prob
        )
        return corrupted_mask

    def embed_state_corrupted(self, s: StateType) -> np.ndarray:
        x, y = self.get_obs_coords(s)
        x, y = self._location_corruption(x, y)
        grid = self._embed_one_hot(x, y)
        grid += self._random_embedding_noise()

        return self._make_embedding_from_grid(grid)

    def _discretize_observation(self, obs: np.ndarray):
        # normalize to the discrete range
        obs = (obs - obs.min()) / (obs.max() - obs.min()) * (
            self.discrete_range[1] - self.discrete_range[0]
        ) + self.discrete_range[0]

        return obs.astype(int)

    def embed_state_discrete(self, s: StateType) -> np.ndarray:
        return self._discretize_observation(self.embed_state_corrupted(s))

    def __call__(self, s: Union[int, List[int]]) -> Union[np.ndarray, List[np.ndarray]]:
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

    action_keys = ["up", "down", "left", "right"]

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
    ) -> np.ndarray:
        states = np.arange(h * w).reshape(h, w)
        transitions = [
            np.vstack([states[0, :], states[:-1, :]]),
            np.vstack([states[1:, :], states[-1, :]]),
            np.hstack([states[:, :1], states[:, :-1]]),
            np.hstack([states[:, 1:], states[:, -1:]]),
        ]
        transitions = [one_hot(t.reshape(-1), h * w) for t in transitions]

        if walls is not None:

            def filter_transition_function(t):
                def _filter_wall(b1, b2):
                    t[b1][b1], t[b1][b2] = t[b1][b1] + t[b1][b2], 0
                    t[b2][b2], t[b2][b1] = t[b2][b2] + t[b2][b1], 0

                for b1, b2 in walls:
                    _filter_wall(b1, b2)
                return t

            transitions = [filter_transition_function(t) for t in transitions]

        return np.array(transitions)

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
        assert action < self.state_action_transitions.shape[0], f"action {action}"
        return self.state_action_transitions[action, state, :]

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

        if self.walls is None:
            return ax
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
        successor_state_rew: Dict[StateType, RewType],
        movement_penalty: float = 0.0,
    ) -> None:
        self.successor_state_rew = successor_state_rew
        self.movement_penalty = movement_penalty

    def get_reward(self, state: StateType):
        return self.successor_state_rew.get(state, self.movement_penalty)

    def get_rew_range(self) -> tuple[RewType, RewType]:
        rews = self.successor_state_rew.values()
        return tuple([min(rews), max(rews)])

    def construct_rew_func(self, transition_function: np.ndarray) -> np.ndarray:
        """
        transition_function -> [n_actions, n_states, n_states]
        returns: a reward function with dimensions [n_states, n_actions]
        """
        _, n_s, _ = transition_function.shape
        reward_function = np.zeros(n_s)
        for s, r in self.successor_state_rew.items():
            reward_function[s] = r
        return np.matmul(transition_function, reward_function)


class GridWorldEnv(gym.Env):
    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        observation_model: ObservationModel,
        initial_state: Optional[int] = None,
        n_states: Optional[int] = None,
        end_state: Optional[list[int]] = None,
        random_initial_state: bool = True,
        max_steps: Optional[int] = None,
        **kwargs,
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

        self.observation = self.generate_observation(self.current_state)
        self.states = np.arange(n_states)
        self.end_state = end_state
        self.step_counter = 0
        self.max_steps = max_steps

        # attributes for gym.Env
        # See: https://gymnasium.farama.org/api/env/
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.observation_model.map_height,
                self.observation_model.map_height,
            ),
            dtype=np.int32,
        )
        self.metadata = None
        self.render_mode = None
        self.reward_range = self.reward_model.get_rew_range()
        self.spec = None

    def _check_terminate(self, state: int) -> bool:
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            return True
        if self.end_state == None:
            return False
        return state in self.end_state

    def _get_initial_state(self) -> int:
        if self.initial_state:
            return self.initial_state
        return np.random.randint(self.n_states)

    def display_gridworld(
        self, ax: Optional[matplotlib.axes.Axes] = None, wall_color="k"
    ) -> matplotlib.axes.Axes:
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))
            ax.invert_yaxis()
        self.transition_model.display_gridworld(ax, wall_color)

        for s, rew in self.reward_model.successor_state_rew.items():
            loc = self.observation_model.get_grid_coords(s)
            c = "b" if rew > 0 else "r"
            ax.annotate(f"{rew}", loc, ha="center", va="center", c=c)
        ax.set_title("Thread-the-needle states")
        return ax

    def generate_observation(self, state: int) -> np.ndarray:
        return self.observation_model(state)

    def get_obseservation(self) -> np.ndarray:
        return self.observation

    def get_state(self) -> np.ndarray:
        return self.current_state

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def get_optimal_policy(self) -> np.ndarray:
        t = self.transition_model.state_action_transitions
        r = self.reward_model.construct_rew_func(t)
        sa_values, values = ValueIterationNetwork.value_iteration(
            t,
            r,
            self.observation_model.h,
            self.observation_model.w,
            gamma=0.8,
            iterations=1000,
        )
        return (
            np.array([np.isclose(v, v.max()) for v in sa_values], dtype=float),
            values,
        )

    # Key methods from Gymnasium:
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, Dict[str, Any]]:
        self.current_state = self._get_initial_state()
        self.step_counter = 0

        return self.generate_observation(self.current_state), dict()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        self.step_counter += 1

        pdf_s = self.transition_model.get_sucessor_distribution(
            self.current_state, action
        )

        assert np.sum(pdf_s) == 1, (action, self.current_state, pdf_s)
        assert np.all(pdf_s >= 0), print(pdf_s)

        successor_state = np.random.choice(self.states, p=pdf_s)
        sucessor_observation = self.generate_observation(successor_state)

        reward = self.reward_model.get_reward(successor_state)
        terminated = self._check_terminate(successor_state)
        truncated = False
        info = dict(start_state=self.current_state, successor_state=successor_state)

        self.current_state = successor_state
        self.observation = sucessor_observation

        output = tuple([sucessor_observation, reward, terminated, truncated, info])

        return output

    def render(self) -> None:
        raise NotImplementedError

    def close(self):
        pass


class ThreadTheNeedleEnv(GridWorldEnv):
    @classmethod
    def create_env(
        cls,
        height: int,
        width: int,
        map_height: int,
        state_rewards: dict[StateType, RewType],
        observation_kwargs: dict[str, Any],
        movement_penalty: float = 0.0,
        **gridworld_env_kwargs,
    ):
        # Define the transitions for the thread the needle task
        walls = make_thread_the_needle_walls(20)
        transition_model = TransitionModel(height, width, walls)

        observation_model = ObservationModel(
            height, width, map_height, **observation_kwargs
        )

        reward_model = RewardModel(state_rewards, movement_penalty)

        return cls(
            transition_model, reward_model, observation_model, **gridworld_env_kwargs
        )


class OpenEnv(ThreadTheNeedleEnv):
    @classmethod
    def create_env(
        cls,
        height: int,
        width: int,
        map_height: int,
        state_rewards: dict[StateType, RewType],
        observation_kwargs: dict[str, Any],
        movement_penalty: float = 0.0,
        **gridworld_env_kwargs,
    ):
        # Define the transitions for the thread the needle task
        transition_model = TransitionModel(height, width, None)

        observation_model = ObservationModel(
            height, width, map_height, **observation_kwargs
        )

        reward_model = RewardModel(state_rewards, movement_penalty)

        return cls(
            transition_model, reward_model, observation_model, **gridworld_env_kwargs
        )


class CnnWrapper(ThreadTheNeedleEnv):
    def __init__(self, env):
        self.parent = env
        self.parent.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.observation_model.map_height,
                self.observation_model.map_height,
                1,
            ),
            dtype=np.uint8,
        )

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def generate_observation(self, state: int) -> np.ndarray:
        return self.parent.generate_observation(state).reshape(
            self.observation_model.map_height, self.observation_model.map_height, 1
        )


def sample_random_walk_states(
    transition_model: TransitionModel,
    walk_length: int,
    initial_state: Optional[int] = None,
) -> list[StateType]:
    random_walk = []
    if initial_state is not None:
        s = initial_state
    else:
        s = choices(list(transition_model.adjecency_list.keys()))[0]

    random_walk.append(s)
    for _ in range(walk_length):
        s = choices(transition_model.adjecency_list[s])[0]
        random_walk.append(s)

    return random_walk


def sample_random_walk(
    length: int,
    transition_model: TransitionModel,
    observation_model: ObservationModel,
    initial_state: Optional[int] = None,
) -> torch.tensor:
    states = sample_random_walk_states(
        transition_model, length, initial_state=initial_state
    )
    obs = torch.stack([make_tensor(observation_model(s)) for s in states])
    return obs


def sample_states(
    transition_model: TransitionModel,
    n: int,
) -> np.ndarray:
    return np.random.choice(len(transition_model.adjecency_list), n).tolist()
