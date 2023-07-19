from dataclasses import dataclass
from random import choice
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.utils import softmax
from state_inference.gridworld_env import ActType, ObsType, RewType
from state_inference.model.tabular_models import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from state_inference.model.vae import StateVae
from state_inference.utils.pytorch_utils import (
    DEVICE,
    convert_8bit_array_to_float_tensor,
    train,
)
from state_inference.utils.utils import inverse_cmf_sampler

BATCH_SIZE = 64
N_EPOCHS = 10000
OPTIM_KWARGS = dict(lr=3e-4)
GRAD_CLIP = True
GAMMA = 0.8
N_ITER_VALUE_ITERATION = 1000
SOFTMAX_GAIN = 1.0


@dataclass
class OaroTuple:
    obs: ObsType
    a: ActType
    r: RewType
    obsp: ObsType


class RandomAgent:
    def __init__(
        self, task, state_inference_model: StateVae, set_action: Set[ActType]
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.set_action = set_action
        self.cached_oaro_tuples = list()
        # self.cached_obs = list()

    def predict(self, obs: ObsType) -> ActType:
        return choice(list(self.set_action)), None

    def update_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, **kwargs) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        for _ in range(total_timesteps):
            # self.cached_obs.append(obs_prev)

            action, _ = self.predict(obs_prev)
            obs, rew, terminated, _, _ = self.task.step(action)

            self.cached_oaro_tuples.append(
                OaroTuple(obs=obs_prev, a=action, r=rew, obsp=obs)
            )
            assert obs.shape

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
            assert obs_prev.shape
            assert not isinstance(obs_prev, tuple)
        # self.cached_obs.append

        self.update_model()


class ValueIterationAgent(RandomAgent):
    TRANSITION_MODEL_CLASS = TabularStateActionTransitionEstimator
    REWARD_MODEL_CLASS = TabularRewardEstimator

    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        set_action: Set[ActType],
        optim_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
    ) -> None:
        super().__init__(task, state_inference_model, set_action)

        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        self.optim = AdamW(self.state_inference_model.parameters(), **optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.beta = softmax_gain

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.q_values = dict()

    def get_hashed_state(self, obs: int) -> tuple[int]:
        obs = self.preprocess_obs(obs)
        return tuple(*self.state_inference_model.get_state(obs.to(DEVICE)))

    # def predict(self, obs: ObsType) -> tuple[ActType, None]:
    #     s = self.get_hashed_state(obs)
    #     q_values = self.q_values.get(s, None)
    #     if q_values:
    #         pmf = softmax(list(q_values.values()), self.beta)
    #         a0 = inverse_cmf_sampler(pmf)
    #         print(pmf, a0, q_values.keys())
    #         print(list(q_values.keys())[a0])
    #         return inverse_cmf_sampler(q_values), None
    #     return super().predict(obs)

    def preprocess_obs(self, obs: Union[ObsType, List[ObsType]]) -> torch.FloatTensor:
        return convert_8bit_array_to_float_tensor(obs)

    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        obs = [o.obs for o in self.cached_oaro_tuples]

        return DataLoader(
            self.preprocess_obs(obs).to(DEVICE),
            batch_size=batch_size,
            shuffle=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        _ = train(self.state_inference_model, dataloader, self.optim, self.grad_clip)

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self.get_hashed_state(obs.obs)
        sp = self.get_hashed_state(obs.obsp)
        return s, obs.a, obs.r, sp

    def update_model(self) -> None:
        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        print("Update Estimates")
        self.transition_estimator.reset()
        self.reward_estimator.reset()
        for obs in self.cached_oaro_tuples:
            s, a, r, sp = self._get_sars_tuples(obs)
            self.transition_estimator.update(s, a, sp)
            self.reward_estimator.update(sp, r)
        print("Run Value Iteration")
        # use value iteration to estimate the rewards
        self.q_values, _ = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
