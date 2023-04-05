import numpy as np
import torch
from typing import Tuple
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


class GridWorld:
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
        if x >= self.h:
            x = self.h
        if y < 0:
            y = 0
        if y >= self.w:
            y = y

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
