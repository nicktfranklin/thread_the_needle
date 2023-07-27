from dataclasses import dataclass
from random import choice
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
    convert_8bit_to_float,
    train,
)

BATCH_SIZE = 64
N_EPOCHS = 20
OPTIM_KWARGS = dict(lr=3e-4)
GRAD_CLIP = True
GAMMA = 0.99
N_ITER_VALUE_ITERATION = 1000
SOFTMAX_GAIN = 1.0
EPSILON = 0.05


@dataclass
class OaroTuple:
    obs: th.tensor
    a: ActType
    r: RewType
    obsp: th.tensor


class BaselineCompatibleAgent:
    def __init__(
        self, task, state_inference_model: StateVae, set_action: Set[ActType]
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.set_action = set_action
        self.cached_oaro_tuples = list()
        self.cached_obs = list()

    def get_env(self):
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def predict(
        self, obs: ObsType, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        if deterministic:
            return np.array(list(self.set_action)[0]), None
        return choice(list(self.set_action)), None

    def _estimate_batch(self) -> None:
        pass

    def _within_batch_update(self, obs: OaroTuple) -> None:
        pass

    def learn(self, total_timesteps: int, **kwargs) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        for _ in range(total_timesteps):
            self.cached_obs.append(th.tensor(obs_prev))

            action = self.predict(obs_prev)[0][0].item()

            obs, rew, terminated, _, _ = self.task.step(action)
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=th.tensor(obs_prev), a=action, r=rew, obsp=th.tensor(obs)
            )

            self._within_batch_update(obs_tuple)

            self.cached_oaro_tuples.append(obs_tuple)

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)

        self.cached_obs.append(th.tensor(obs))

        self._estimate_batch()


class SoftmaxPolicy:
    def __init__(
        self,
        feature_extractor: StateVae,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.feature_extractor = feature_extractor
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.hash_vector = np.array(
            [
                self.feature_extractor.z_dim**ii
                for ii in range(self.feature_extractor.z_layers)
            ]
        )
        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def _preprocess_obs(self, obs: Union[ObsType, List[ObsType]]) -> th.Tensor:
        return convert_8bit_array_to_float_tensor(obs)

    def _get_hashed_state(self, obs: ObsType) -> tuple[int]:
        obs = self._preprocess_obs(obs)
        z = self.feature_extractor.get_state(obs.to(DEVICE))
        return z.dot(self.hash_vector)

    def maybe_init_q_values(self, s: int) -> None:
        if s not in self.q_values:
            self.q_values[s] = self.q_init

    def get_distribution(self, obs: th.Tensor) -> th.Tensor:
        s = self._get_hashed_state(obs)

        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = th.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = th.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)


class ValueIterationAgent(BaselineCompatibleAgent):
    TRANSITION_MODEL_CLASS = TabularStateActionTransitionEstimator
    REWARD_MODEL_CLASS = TabularRewardEstimator
    POLICY_CLASS = SoftmaxPolicy

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

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            feature_extractor=state_inference_model,
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=len(set_action),
        )
        self.list_states = list()

        assert epsilon >= 0 and epsilon < 1.0

    def _precomput_all_obs(self):
        return convert_8bit_to_float(th.stack(self.cached_obs)).to(DEVICE)

    def _precompute_states(self):
        obs_tensors = self._precomput_all_obs()
        states = self.state_inference_model.get_state(obs_tensors)
        self.list_states = states.dot(self.policy.hash_vector)

    def predict(
        self, obs: ObsType, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None
        p = self.policy.get_distribution(obs)
        return p.get_actions(deterministic=deterministic), None

    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        return DataLoader(
            self._precomput_all_obs(),
            batch_size=batch_size,
            shuffle=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        for _ in range(N_EPOCHS):
            self.state_inference_model.train()
            _ = train(
                self.state_inference_model, dataloader, self.optim, self.grad_clip
            )
            self.state_inference_model.prep_next_batch()

    def _get_hashed_state(self, obs: th.tensor):
        return self.state_inference_model.get_state(convert_8bit_to_float(obs)).dot(
            self.policy.hash_vector
        )

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self._get_hashed_state(obs.obs)[0]
        sp = self._get_hashed_state(obs.obsp)[0]
        return s, obs.a, obs.r, sp

    def _update_rew_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def _estimate_reward_model(self) -> None:
        self.reward_estimator.reset()
        for obs in self.cached_oaro_tuples:
            self._update_rew_model(obs)

    def _estimate_batch(self) -> None:
        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        self.transition_estimator.reset()
        self.reward_estimator.reset()
        # self._precompute_states()  # for speed

        self._estimate_reward_model()

        # use value iteration to estimate the rewards
        self.policy.q_values, value_function = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
        self.value_function = value_function


class ViAgentWithExploration(ValueIterationAgent):
    eta = 0.1

    def _within_batch_update(self, obs: OaroTuple) -> None:
        s, a, r, sp = self._get_sars_tuples(obs)

        self.policy.maybe_init_q_values(s)
        q_s_a = self.policy.q_values[s][a]

        self.policy.maybe_init_q_values(sp)
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a = (1 - self.eta) * q_s_a + self.eta * (r + self.gamma * V_sp)
        self.policy.q_values[s][a] = q_s_a
