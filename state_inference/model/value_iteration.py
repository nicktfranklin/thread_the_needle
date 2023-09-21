from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

import state_inference.model.vae
from state_inference.model.common import (
    BaselineCompatibleAgent,
    OaroTuple,
    SoftmaxPolicy,
)
from state_inference.model.tabular_models import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from state_inference.model.vae import StateVae
from state_inference.utils.pytorch_utils import DEVICE, convert_8bit_to_float, train


class ValueIterationAgent(BaselineCompatibleAgent):
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
        n_epochs: int = 20,
        alpha: float = 0.05,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(task, state_inference_model)

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

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(state_inference.model.vae, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])

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

    def _prep_vae_dataloader(self, batch_size: int):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
        """
        obs = self.rollout_buffer.get_tensor("obs")
        obs = convert_8bit_to_float(obs).to(DEVICE)
        obs = obs.permute(0, 3, 1, 2)  # -> NxCxHxW

        return DataLoader(
            obs,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _train_vae_batch(self, progress_bar: Optional[bool] = False):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        if progress_bar:
            iterator = trange(self.n_epochs, desc="Vae Batches")
        else:
            iterator = range(self.n_epochs)

        for _ in iterator:
            self.state_inference_model.train()
            train(self.state_inference_model, dataloader, self.optim, self.grad_clip)
            self.state_inference_model.prep_next_batch()

        # clear memory
        torch.cuda.empty_cache()

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

    def update_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        # pass all of the observations throught the model (really, twice)
        # for speed. It's much faster in a batch than single observations
        obs = self.rollout_buffer.get_tensor("obs")
        obsp = self.rollout_buffer.get_tensor("obsp")
        s = self._get_hashed_state(obs)
        sp = self._get_hashed_state(obsp)
        return s, sp

    def retrain_model(self):
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        s, sp = self._precalculate_states_for_batch_training()

        for idx, obs in enumerate(self.rollout_buffer.get_all()):
            self.transition_estimator.update(s[idx], obs.a, sp[idx])
            self.reward_estimator.update(sp[idx], obs.r)

    def _batch_estimate(
        self, step: int, last_step: bool, progress_bar: Optional[bool] = False
    ) -> None:
        # update only periodically
        if (not last_step) and (
            self.n_steps is None or step == 0 or step % self.n_steps != 0
        ):
            return

        # update the state model
        self._train_vae_batch(progress_bar=progress_bar)

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

    def update_qvalues(self, s, a, r, sp) -> None:
        self.policy.maybe_init_q_values(s)
        q_s_a = self.policy.q_values[s][a]

        self.policy.maybe_init_q_values(sp)
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a = (1 - self.alpha) * q_s_a + self.alpha * (r + self.gamma * V_sp)
        self.policy.q_values[s][a] = q_s_a

    def _within_batch_update(
        self, obs: OaroTuple, state: Optional[Tensor], state_prev: Optional[Tensor]
    ) -> None:
        # state and state_prev are only used by recurrent models
        assert state is None
        assert state_prev is None

        s, a, r, sp = self._get_sars_tuples(obs)
        self.update_qvalues(s, a, r, sp)
