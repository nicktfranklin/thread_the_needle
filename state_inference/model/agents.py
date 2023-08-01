from dataclasses import dataclass
from random import choice
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
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
from state_inference.utils.data import RecurrentVaeDataset, TransitionVaeDataset
from state_inference.utils.pytorch_utils import DEVICE, convert_8bit_to_float, train

BATCH_SIZE = 64
N_EPOCHS = 20
MAX_SEQUENCE_LEN = 10
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


class BaseAgent:
    def __init__(
        self, task, state_inference_model: StateVae, set_action: Set[ActType]
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.set_action = set_action
        self.cached_obs = list()

    def get_env(self):
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            return np.array(list(self.set_action)[0]), None
        return choice(list(self.set_action)), None

    def _estimate_batch(self) -> None:
        pass

    def _within_batch_update(self, obs: OaroTuple) -> None:
        pass

    def _init_state(self):
        return None

    def learn(
        self, total_timesteps: int, estimate_batch: bool = True, **kwargs
    ) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        episode_start = True
        state = self._init_state()

        for _ in range(total_timesteps):
            action, state = self.predict(obs_prev, state, episode_start)
            episode_start = False

            obs, rew, terminated, _, _ = self.task.step(action.item())
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=th.tensor(obs_prev), a=action.item(), r=rew, obsp=th.tensor(obs)
            )

            self._within_batch_update(obs_tuple)

            self.cached_obs.append(obs_tuple)

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)
        if estimate_batch:
            self._estimate_batch()


class SoftmaxPolicy:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        if s not in self.q_values:
            self.q_values[s] = self.q_init

    def get_distribution(self, s: int) -> th.Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = th.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = th.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)


class ValueIterationAgent(BaseAgent):
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
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=len(set_action),
        )
        self.list_states = list()

        assert epsilon >= 0 and epsilon < 1.0

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

    def _precomput_all_obs(self):
        return convert_8bit_to_float(th.stack([obs.obs for obs in self.cached_obs])).to(
            DEVICE
        )

    def _precompute_states(self):
        obs_tensors = self._precomput_all_obs()
        states = self.state_inference_model.get_state(obs_tensors)
        self.list_states = states.dot(self.hash_vector)

    def get_policy(self, obs: ObsType):
        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p

    def predict(
        self, obs: ObsType, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None

        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
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
            train(self.state_inference_model, dataloader, self.optim, self.grad_clip)
            self.state_inference_model.prep_next_batch()

    def _get_hashed_state(self, obs: th.tensor):
        obs = obs if isinstance(obs, th.Tensor) else th.tensor(obs)
        return self.state_inference_model.get_state(convert_8bit_to_float(obs)).dot(
            self.hash_vector
        )

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self._get_hashed_state(obs.obs)[0]
        sp = self._get_hashed_state(obs.obsp)[0]
        return s, obs.a, obs.r, sp

    def update_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def _estimate_batch(self) -> None:
        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        for obs in self.cached_obs:
            self.update_model(obs)

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

    def update_qvalues(self, obs: OaroTuple) -> None:
        s, a, r, sp = self._get_sars_tuples(obs)

        self.policy.maybe_init_q_values(s)
        q_s_a = self.policy.q_values[s][a]

        self.policy.maybe_init_q_values(sp)
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a = (1 - self.eta) * q_s_a + self.eta * (r + self.gamma * V_sp)
        self.policy.q_values[s][a] = q_s_a

    def _within_batch_update(self, obs: OaroTuple) -> None:
        self.update_qvalues(obs)


class ViControlableStateInf(ViAgentWithExploration):
    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        obs = th.stack([obs.obs for obs in self.cached_obs])
        obsp = th.stack([obs.obsp for obs in self.cached_obs])
        a = F.one_hot(
            th.tensor([obs.a for obs in self.cached_obs]), num_classes=4
        ).float()

        dataset = TransitionVaeDataset(obs, a, obsp)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )


class ViDynaAgent(ViControlableStateInf):
    k = 10

    def _within_batch_update(self, obs: OaroTuple) -> None:
        # update the current q-value
        self.update_qvalues(obs)

        # dyna updates (note: this assumes a deterministic enviornment,
        # and this code differes from dyna as we are only using resampled
        # values and not seperately sampling rewards and sucessor states
        if self.cached_obs:
            for _ in range(self.k):
                obs = choice(self.cached_obs)
                self.update_qvalues(obs)


class RecurrentStateInf(ViAgentWithExploration):
    def _prep_vae_dataloader(
        self, batch_size: int = BATCH_SIZE, max_sequence_len: int = MAX_SEQUENCE_LEN
    ):
        obs = th.stack([obs.obs for obs in self.cached_obs])
        actions = F.one_hot(th.tensor([obs.a for obs in self.cached_obs]))
        return RecurrentVaeDataset.contruct_dataloader(
            obs, actions, max_sequence_len, batch_size
        )

    def _init_state(self):
        return super()._init_state()

    def predict(
        self,
        obs: ObsType,
        state: th.Tensor,
        episode_start=None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None

        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic), None
