from collections import namedtuple
from random import choices
from typing import Dict, Hashable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from scipy.signal import fftconvolve


# The state-location is smoothed by convoling an RBF kernel with the
# one-hot representation of the location in the nXn Grid
class RbfKernelEmbedding:
    def __init__(self, kernel_size: int = 51, len_scale: float = 0.15) -> None:
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.kernel = torch.empty([kernel_size, kernel_size]).float()
        center = torch.ones([1, 1]) * kernel_size // 2

        def _rbf(x: torch.tensor) -> float:
            dist = ((x - center) ** 2).sum() ** 0.5
            return torch.exp(-len_scale * dist)

        for r in range(kernel_size):
            for c in range(kernel_size):
                self.kernel[r, c] = _rbf(torch.tensor([r, c]))

    def __call__(self, input_space: torch.tensor) -> torch.tensor:
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

    def get_obs_coords(self, s: int) -> Tuple[int, int]:
        return self.states[s]

    def get_grid_location(self, s: int) -> np.ndarray:
        raw_state = np.zeros((self.h, self.w))
        raw_state[0, s] = 1
        return raw_state

    def _embed_one_hot(self, x: int, y: int) -> torch.tensor:
        grid = torch.zeros((self.map_height, self.map_height))
        grid[x, y] = 1.0
        return grid

    def embed_state(self, s: int) -> torch.tensor:
        x, y = self.get_obs_coords(s)
        grid = self._embed_one_hot(x, y)
        return self.kernel(grid)

    def _location_corruption(self, x: int, y: int) -> Tuple[int, int]:
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

    def _random_embedding_noise(self) -> torch.tensor:
        corrupted_mask = torch.exp(
            torch.randn(self.map_height, self.map_height) * self.noise_log_scale
            + self.noise_log_mean
        )
        corrupted_mask *= (
            torch.rand(self.map_height, self.map_height) < self.noise_corruption_prob
        )
        return corrupted_mask

    def embed_state_corrupted(self, s: int) -> torch.tensor:
        x, y = self.get_obs_coords(s)
        x, y = self._location_corruption(x, y)
        grid = self._embed_one_hot(x, y)
        grid += self._random_embedding_noise()
        return self.kernel(grid)

    def __call__(self, s: int) -> torch.tensor:
        return self.embed_state_corrupted(s)

    def display_state(self, s: int) -> None:
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
    # Generative model. Assumes connectivity between neighboring states

    def __init__(
        self, h: int, w: int, walls: Optional[List[Tuple[int, int]]] = None
    ) -> None:
        self.transitions = self._make_transitions(h, w)
        self.edges = self._make_edges(self.transitions)
        if walls:
            self._add_walls(walls)

        self.n_states = h * w

    @staticmethod
    def _make_transitions(h: int, w: int) -> np.ndarray:
        t = np.zeros((h * w, h * w))
        for s0 in range(h * w):
            # if s0 + 1

            if s0 - w >= 0:
                t[s0, s0 - w] = 1
            if s0 + w + 1 < h * w:
                t[s0, s0 + w] = 1

            if (s0 + 1) % w > 0:
                t[s0, s0 + 1] = 1

            if s0 % w > 0:
                t[s0, s0 - 1] = 1

        # normalize
        return TransitionModel._normalize(t)

    @staticmethod
    def _normalize(t: np.ndarray) -> np.ndarray:
        return t / np.tile(t.sum(axis=1).reshape(-1, 1), t.shape[1])

    @staticmethod
    def _make_edges(transitions: np.ndarray) -> Dict[int, np.ndarray]:
        edges = {}
        for s, t in enumerate(transitions):
            edges[s] = np.where(t > 0)[0]
        return edges

    def _add_walls(self, walls: List[Tuple[int, int]]) -> None:
        for s, sp in walls:
            self.transitions[s, sp] = 0
        self.transitions = self._normalize(self.transitions)

    def generate_random_walk(self, walk_length: int) -> Tuple[np.ndarray, List[int]]:
        random_walk = []
        s = choices(list(self.edges.keys()))[0]
        random_walk.append(s)
        state_counts = np.zeros(len(self.edges))
        for _ in range(walk_length):
            s = choices(self.edges[s])[0]
            state_counts[s] += 1
            random_walk.append(s)

        return state_counts, random_walk

    def sample_states(self, n: int, kind: str = "walk") -> np.ndarray:
        if kind.lower() == "random":

            def state_sampler(n):
                return np.random.choice(len(self.edges), n)

        elif kind.lower() == "walk":

            def state_sampler(n):
                return self.generate_random_walk(n - 1)[1]

        else:
            raise NotImplementedError("only type 'walk' or 'random' are implemented")
        return state_sampler(n)


class RewardModel:
    # Generative Model
    pass


class TransitionEstimator:
    ## Note: does not take in actions

    def __init__(self):
        self.transitions = dict()
        self.pmf = dict()

    def update(self, s: Hashable, sp: Hashable):
        if s in self.transitions:
            if sp in self.transitions[s]:
                self.transitions[s][sp] += 1
            else:
                self.transitions[s][sp] = 1
        else:
            self.transitions[s] = {sp: 1}

        N = float(sum(self.transitions[s].values()))
        self.pmf[s] = {sp0: v / N for sp0, v in self.transitions[s].items()}

    def batch_update(self, list_states: List[Hashable]):
        for ii in range(len(list_states) - 1):
            self.update(list_states[ii], list_states[ii + 1])

    def get_transition_probs(self, s: Hashable) -> Dict[Hashable, float]:
        # if a state is not in the model, assume it's self-absorbing
        if s not in self.pmf:
            return {s: 1.0}
        return self.pmf[s]


class RewardEstimator:
    def __init__(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():
            self.counts[s] += np.array([r, 1])
        else:
            self.counts[s] = np.array([r, 1])

        self.state_reward_function[s] = self.counts[s][0] / self.counts[s][1]

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_reward(self, state):
        return self.state_reward_function[state]


def value_iteration(
    t: Dict[Union[str, int], TransitionEstimator],
    r: RewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _inner_sum(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * v[sp]
        return _sum

    def _expected_reward(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * r.get_reward(sp)
        return _sum

    for k in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = _expected_reward(s, a) + gamma * _inner_sum(s, a)
        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v


### define simple deterministic transition functions using cardinal movements
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def make_cardinal_transition_function(h: int, w: int) -> Dict[str, np.ndarray]:
    states = np.arange(h * w).reshape(h, w)
    transitions = {
        "up": np.vstack([states[0, :], states[:-1, :]]),
        "down": np.vstack([states[1:, :], states[-1, :]]),
        "left": np.hstack([states[:, :1], states[:, :-1]]),
        "right": np.hstack([states[:, 1:], states[:, -1:]]),
    }
    transitions = {k: one_hot(v.reshape(-1), h * w) for k, v in transitions.items()}
    return transitions


# ~~~~~~ UNUSED CODE BELOW HERE ~~~~~~

OaorTuple = namedtuple("OAORTuple", ["o", "a", "op", "r"])


class WorldModel:
    def __init__(
        self,
        transition_functions: Dict[Union[str, int], np.ndarray],
        state_reward_function: Dict[Union[str, int], float],
        observation_model: ObservationModel,
        initial_state: Optional[int] = None,
        n_states: Optional[int] = None,
    ) -> None:
        self.t = transition_functions
        self.r = state_reward_function
        self.observation_model = observation_model

        if not n_states:
            n_states = self.t[0].shape[0]
        if not initial_state:
            initial_state = np.random.randint(n_states)

        self.n_states = n_states
        self.initial_state = initial_state
        self.state = initial_state
        self.observation = self._generate_observation(self.state)

        self.states = np.arange(n_states)

    def _generate_observation(self, state: int) -> torch.tensor:
        return self.observation_model(state).reshape(-1)

    def get_obseservation(self) -> torch.tensor:
        return self.observation

    def take_action(self, action: Union[str, int]) -> OaorTuple:
        assert action in self.t.keys()

        ta = self.t[action]
        assert np.sum(ta) == 1
        assert np.all(ta >= 0)

        sucessor_state = np.choice(self.states, 1, p=ta)
        sucessor_observation = self._generate_observation(sucessor_state)

        obs_tuple = OaorTuple(
            self.get_obseservation(),
            action,
            sucessor_observation,
            self.r[sucessor_state],
        )

        self.state = sucessor_state
        self.observation = sucessor_observation

        return obs_tuple
