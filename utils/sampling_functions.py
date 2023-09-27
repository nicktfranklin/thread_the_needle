from random import choices
from typing import Optional

import numpy as np
import torch

from task.observation_model import ObservationModel
from task.transition_model import TransitionModel
from task.utils import StateType
from utils.pytorch_utils import make_tensor


def sample_states(transition_model: TransitionModel, n) -> np.ndarray:
    return np.random.choice(len(transition_model.adjecency_list), n).tolist()


def sample_random_walk_states(
    transition_model: TransitionModel,
    walk_length: int,
    initial_state: Optional[StateType] = None,
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
    initial_state: Optional[StateType] = None,
) -> torch.tensor:
    states = sample_random_walk_states(
        transition_model, length, initial_state=initial_state
    )
    obs = torch.stack([make_tensor(observation_model(s)) for s in states])
    return obs
