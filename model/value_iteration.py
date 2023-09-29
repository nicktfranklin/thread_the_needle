from random import choice
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import Tensor
from tqdm import trange

import model.vae
from model.common import OaroTuple, RolloutBuffer, SoftmaxPolicy
from model.tabular_models import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from model.vae import StateVae
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class ValueIterationAgent:
    TRANSITION_MODEL_CLASS = TabularStateActionTransitionEstimator
    REWARD_MODEL_CLASS = TabularRewardEstimator
    POLICY_CLASS = SoftmaxPolicy
    """
    Value iteration agent. Collects rollouts using Q-learning with an optimistic exploration
    policy on a state-inference model (VAE) and then updates the state-inference model with the
    roll outs. The Value-iteration over the rollouts are used to re-estimate state-action values.

    :param state_inference_model: The VAE used to estimate the State

    """

    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip: bool = True,
        batch_size: int = 64,
        gamma: float = 0.99,
        n_iter: int = 1000,
        softmax_gain: float = 1.0,
        epsilon: float = 0.05,
        n_steps: Optional[int] = None,  # None => only update at the end,
        n_epochs: int = 10,
        alpha: float = 0.05,
        dyna_updates: int = 5,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.rollout_buffer = RolloutBuffer()
        self.allobs = list()

        self.optim = self.state_inference_model.configure_optimizers(optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.alpha = alpha

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=task.action_space.n,
        )

        assert epsilon >= 0 and epsilon < 1.0

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

        self.num_timesteps = 0
        self.value_function = None
        self.n_dyna_updates = dyna_updates

    def _init_state(self):
        return None

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def _get_hashed_state(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self._get_hashed_state(obs.obs)[0]
        sp = self._get_hashed_state(obs.obsp)[0]
        return s, obs.a, obs.r, sp

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        # pass all of the observations throught the model (really, twice)
        # for speed. It's much faster in a batch than single observations
        obs = self.rollout_buffer.get_tensor("obs")
        obsp = self.rollout_buffer.get_tensor("obsp")
        s = self._get_hashed_state(obs)
        sp = self._get_hashed_state(obsp)
        return s, sp

    def _dyna_updates(self, n_updates: int) -> None:
        #### Dyna updates ####
        # dyna updates (note: this assumes a deterministic enviornment,
        # and this code differes from dyna as we are only using resampled
        # values and not seperately sampling rewards and sucessor states
        if self.rollout_buffer.len() > 0:
            for _ in range(n_updates):
                # select a random observation from experience
                obs = choice(self.rollout_buffer.get_all())
                s = self._get_hashed_state(obs.obs)[0]
                a = obs.a

                # draw r, sp from the model
                sp = self.transition_estimator.sample(s, a)
                r = self.reward_estimator.sample(sp)

                self.update_qvalues(s, a, r, sp)

    def _update_rollout_policy(
        self, obs: OaroTuple, state: Optional[Tensor], state_prev: Optional[Tensor]
    ) -> None:
        # the rollout policy is DYNA

        # state and state_prev are only used by recurrent models
        assert state is None
        assert state_prev is None

        # pass the obseration tuple through the state-inference network
        s, a, r, sp = self._get_sars_tuples(obs)

        # update the model
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

        # update q-values
        self.update_qvalues(s, a, r, sp)

        self._dyna_updates(self.n_dyna_updates)

    def _init_index(self):
        if self.rollout_buffer.len() == 0:
            return 0
        last_obs = self.rollout_buffer.get_all()[-1]
        return last_obs.index + 1

    def get_env(self):
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def get_policy(self, obs: Tensor):
        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[Tensor, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None

        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic), None

    def update_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def update_qvalues(self, s, a, r, sp) -> None:
        """
        This is the Dyna-Q update rule. From Sutton and Barto (2020), ch 8
        """

        self.policy.maybe_init_q_values(s)
        self.policy.maybe_init_q_values(sp)

        q_s_a = self.policy.q_values[s][a]
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a += self.alpha * (r + self.gamma * V_sp - q_s_a)
        self.policy.q_values[s][a] = q_s_a

    def collect_rollouts(self, n_rollout_steps, progress_bar=None, eval_only=False):
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        episode_start = True
        state_prev = self._init_state()
        idx = self._init_index()

        if progress_bar is not None:
            progress_bar.set_description("Collecting Rollouts")

        self.rollout_buffer.reset()

        n_steps = 0
        while n_steps < n_rollout_steps:
            if progress_bar is not None:
                progress_bar.update(1)
            action, state = self.predict(obs_prev, state_prev, episode_start)
            episode_start = False

            obs, rew, terminated, _, _ = self.task.step(action.item())
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=torch.tensor(obs_prev),
                a=action.item(),
                r=rew,
                obsp=torch.tensor(obs),
                index=idx,
            )

            if not eval_only:
                self._update_rollout_policy(obs_tuple, state, state_prev)

            self.rollout_buffer.add(obs_tuple)

            n_steps += 1
            self.num_timesteps += 1

            obs_prev = obs
            state_prev = state
            if terminated:
                obs_prev = self.task.reset()[0]
                idx += 1
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)
        return

    def estimate_offline(self):
        # resetimate the model from the new states
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        s, sp = self._precalculate_states_for_batch_training()

        for idx, obs in enumerate(self.rollout_buffer.get_all()):
            self.transition_estimator.update(s[idx], obs.a, sp[idx])
            self.reward_estimator.update(sp[idx], obs.r)

        # use value iteration to estimate the rewards
        self.policy.q_values, value_function = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
        self.value_function = value_function

    def learn(
        self,
        total_timesteps: int,
        eval_only: bool = False,
        progress_bar: bool = False,
        **kwargs,
    ) -> None:
        if progress_bar is not None:
            progress_bar = trange(total_timesteps, position=0, leave=True)

        # alternate between collecting rollouts and batch updates
        n_rollout_steps = self.n_steps if self.n_steps is not None else total_timesteps
        while self.num_timesteps < total_timesteps:
            # collect rollouts to train the VAE
            self.collect_rollouts(n_rollout_steps, progress_bar, eval_only)

            if eval_only:
                continue

            if progress_bar is not None:
                progress_bar.set_description("Updating Batch")

            # train the VAE on the rollouts
            dataloader = self.rollout_buffer.get_vae_dataloader(self.batch_size)
            self.state_inference_model.train_epochs(
                self.n_epochs,
                dataloader,
                self.optim,
                self.grad_clip,
                progress_bar=False,
            )

            # clear memory
            torch.cuda.empty_cache()

            # re-estimate the model with the new steps
            self.estimate_offline()

        if progress_bar is not None:
            progress_bar.close()

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(model.vae, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])
