from dataclasses import dataclass
from random import choice
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange

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
BATCH_LENGTH = 10000


@dataclass
class OaroTuple:
    obs: Tensor
    a: ActType
    r: RewType
    obsp: Tensor


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

    def _batch_estimate(self, step: int, last_step: bool) -> None:
        pass

    def _within_batch_update(self, obs: OaroTuple) -> None:
        pass

    def _init_state(self):
        return None

    def learn(
        self,
        total_timesteps: int,
        estimate_batch: bool = True,
        progress_bar: bool = False,
        **kwargs,
    ) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        episode_start = True
        state = self._init_state()

        if progress_bar:
            iterator = trange(total_timesteps)
        else:
            iterator = range(total_timesteps)
        for step in iterator:
            action, state = self.predict(obs_prev, state, episode_start)
            episode_start = False

            obs, rew, terminated, _, _ = self.task.step(action.item())
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=torch.tensor(obs_prev),
                a=action.item(),
                r=rew,
                obsp=torch.tensor(obs),
            )

            self._within_batch_update(obs_tuple)

            self.cached_obs.append(obs_tuple)

            obs_prev = obs
            if terminated:
                obs_prev = self.task.reset()[0]
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)

            if estimate_batch:
                self._batch_estimate(step, step == total_timesteps - 1)


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
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init
            self.q_values[s] = q_init

    def get_distribution(self, s: int) -> Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = torch.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = torch.stack([_get_q(s0) for s0 in s])
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
        batch_length: int = BATCH_LENGTH,
    ) -> None:
        super().__init__(task, state_inference_model, set_action)

        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        self.optim = AdamW(self.state_inference_model.parameters(), **optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.batch_length = batch_length

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=len(set_action),
        )

        assert epsilon >= 0 and epsilon < 1.0

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

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

    def _prep_vae_dataloader(self, batch_size: int, n_trailing: int):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
            n_trailing (int): the number of trailing observations to select
        """
        obs = torch.stack([o.obs for o in self.cached_obs[-n_trailing:]])
        obs = convert_8bit_to_float(obs).to(DEVICE)

        return DataLoader(
            obs,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size, self.batch_length)
        for _ in range(N_EPOCHS):
            self.state_inference_model.train()
            train(self.state_inference_model, dataloader, self.optim, self.grad_clip)
            self.state_inference_model.prep_next_batch()

    def _get_hashed_state(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
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

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        # pass all of the observations throught the model (really, twice)
        # for speed. It's much faster in a batch than single observations
        obs = torch.stack([o.obs for o in self.cached_obs])
        obsp = torch.stack([o.obsp for o in self.cached_obs])
        s = self._get_hashed_state(obs)
        sp = self._get_hashed_state(obsp)
        return s, sp

    def retrain_model(self):
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        s, sp = self._precalculate_states_for_batch_training()

        for idx, obs in enumerate(self.cached_obs):
            self.transition_estimator.update(s[idx], obs.a, sp[idx])
            self.reward_estimator.update(sp[idx], obs.r)

    def _batch_estimate(self, step: int, last_step: bool) -> None:
        # update only periodically
        if (not last_step) and (step == 0 or step % self.batch_length != 0):
            return

        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        self.retrain_model()

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
        obs = torch.stack([obs.obs for obs in self.cached_obs])
        obsp = torch.stack([obs.obsp for obs in self.cached_obs])
        a = F.one_hot(
            torch.tensor([obs.a for obs in self.cached_obs]), num_classes=4
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
    max_sequence_len: int = MAX_SEQUENCE_LEN

    def _prep_vae_dataloader(
        self,
        batch_size: int,
        n_trailing_obs: int,
    ):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
            n_trailing (int): the number of trailing observations to select
        """
        obs = torch.stack([obs.obs for obs in self.cached_obs[-n_trailing_obs:]])
        actions = F.one_hot(torch.tensor([obs.a for obs in self.cached_obs]))
        return RecurrentVaeDataset.contruct_dataloader(
            obs, actions, self.max_sequence_len, batch_size
        )

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _init_state(self):
        # unbatch state for the agent (not vae training)
        return torch.zeros(
            self.state_inference_model.z_dim * self.state_inference_model.z_layers
        )

    def _get_hashed_state(self, obs: Tensor, state_prev: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs = convert_8bit_to_float(obs)

        assert isinstance(state_prev, Tensor)

        state = self.state_inference_model.get_state(obs, state_prev)
        return state.dot(self.hash_vector)

    def predict(
        self,
        obs: ObsType,
        state: Tensor,
        episode_start=None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None

        s = self._get_hashed_state(obs, state)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic), None
