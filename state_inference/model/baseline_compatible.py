from dataclasses import dataclass
from random import choice
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.utils import softmax
from state_inference.gridworld_env import ActType, GridWorldEnv, ObsType, RewType
from state_inference.model.tabular_models import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from state_inference.model.vae import StateVae
from state_inference.utils.pytorch_utils import (
    DEVICE,
    convert_8bit_array_to_float_tensor,
    convert_8bit_to_float,
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
EPSILON = 0.05


@dataclass
class OaroTuple:
    obs: int
    a: ActType
    r: RewType
    obsp: int


class EnvWrapper:
    ### for compatibility with the stable baseline codebase

    def __init__(self, task: GridWorldEnv) -> None:
        self.task = task

    def __getattr__(self, attr):
        return getattr(self.task, attr)

    def env_method(self, method_name: str, method_args):
        f = getattr(self.task, method_name)
        return f(method_args)


class BaselineCompatibleAgent:
    def __init__(
        self, task, state_inference_model: StateVae, set_action: Set[ActType]
    ) -> None:
        self.task: GridWorldEnv = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.set_action = set_action
        self.cached_oaro_tuples = list()
        self.cached_obs = list()

    def get_env(self):
        return EnvWrapper(self.task)

    def predict(
        self, obs: ObsType, deterministic: bool = False
    ) -> tuple[ActType, None]:
        if deterministic:
            return list(self.set_action)[0]
        return choice(list(self.set_action)), None

    def update_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, **kwargs) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        for t in range(total_timesteps):
            self.cached_obs.append(torch.tensor(obs_prev))

            action, _ = self.predict(obs_prev)
            obs, rew, terminated, _, _ = self.task.step(action)

            self.cached_oaro_tuples.append(
                OaroTuple(obs=t, a=action, r=rew, obsp=t + 1)
            )
            assert obs.shape

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
            assert obs_prev.shape
            assert not isinstance(obs_prev, tuple)

        self.cached_obs.append(torch.tensor(obs))

        self.update_model()


class ValueIterationAgent(BaselineCompatibleAgent):
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
        epsilon: float = EPSILON,
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
        self.list_states = list()

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )
        assert epsilon >= 0 and epsilon < 1.0
        self.epsilon = epsilon

    def _precompute_states(self):
        obs_tensors = self._precomput_all_obs()
        states = self.state_inference_model.get_state(obs_tensors)
        self.list_states = states.dot(self.hash_vector)

    def get_hashed_state(self, obs: ObsType) -> tuple[int]:
        obs = self.preprocess_obs(obs)
        z = self.state_inference_model.get_state(obs.to(DEVICE))
        return z.dot(self.hash_vector)[0]

    def predict(
        self, obs: ObsType, deterministic: bool = False
    ) -> tuple[ActType, None]:
        s = self.get_hashed_state(obs)

        q_values = self.q_values.get(s, None)
        if q_values:
            pmf = softmax(np.array(list(q_values.values())), self.beta)
            pmf = pmf * (1 - self.epsilon) + (
                self.epsilon / self.transition_estimator.n_actions
            )
            if deterministic:
                return np.argmax(pmf), None
            a0 = inverse_cmf_sampler(pmf)
            return inverse_cmf_sampler(pmf), None
        return super().predict(obs, deterministic)

    def preprocess_obs(self, obs: Union[ObsType, List[ObsType]]) -> torch.FloatTensor:
        return convert_8bit_array_to_float_tensor(obs)

    def _precomput_all_obs(self):
        return convert_8bit_to_float(torch.stack(self.cached_obs)).to(DEVICE)

    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        return DataLoader(
            self._precomput_all_obs(),
            batch_size=batch_size,
            shuffle=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        _ = train(self.state_inference_model, dataloader, self.optim, self.grad_clip)

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self.list_states[obs.obs]
        sp = self.list_states[obs.obsp]
        return s, obs.a, obs.r, sp

    def update_model(self) -> None:
        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        self.transition_estimator.reset()
        self.reward_estimator.reset()
        self._precompute_states()  # for speed

        for obs in self.cached_oaro_tuples:
            s, a, r, sp = self._get_sars_tuples(obs)
            self.transition_estimator.update(s, a, sp)
            self.reward_estimator.update(sp, r)

        # use value iteration to estimate the rewards
        self.q_values, _ = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
