from random import choices
from typing import Optional, Tuple

import numpy as np
import torch

from task.gridworld import StateType
from task.observation_model import ObservationModel
from task.transition_model import TransitionModel
from utils.pytorch_utils import make_tensor


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


def get_state_from_position(row: int, column: int, n_columns: int) -> int:
    return n_columns * row + column


def get_position_from_state(state: int, n_columns: int) -> Tuple[int, int]:
    row = state // n_columns
    column = state % n_columns
    return row, column


def sample_states(
    transition_model: TransitionModel,
    n: int,
) -> np.ndarray:
    return np.random.choice(len(transition_model.adjecency_list), n).tolist()
