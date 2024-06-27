import inspect
from typing import Any, Dict, Hashable, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# from stable_baselines3.common.policies import ActorCriticPolicy
from torch import FloatTensor, Tensor

import model.state_inference.vae
from model.agents.utils.base_agent import BaseAgent
from model.state_inference.nets.mlp import MLP
from model.state_inference.vae import StateVae
from model.training.rollout_data import EpisodeBuffer
from task.utils import ActType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class DiscretePPO(BaseAgent):

    def __init__(
        self,
        env: gym.Env,
        state_inference_model: StateVae,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(env)
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.optim = self._configure_optimizers(optim_kwargs)

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

        input_size = (
            self.state_inference_model.z_dim * self.state_inference_model.z_layers
        )

        self.policy = MLP(
            input_size=input_size,
            hidden_sizes=[input_size],
            output_size=env.action_space.n,
        )

        self.critic = MLP(
            input_size=input_size,
            hidden_sizes=[input_size],
            output_size=1,
        )

    def _init_state(self):
        return None

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def _get_state_hashkey(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def get_policy(self, obs: Tensor):
        """
        assume obs is shape (NxHxWxC).

        Not used for training~
        """
        raise NotImplementedError()

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        raise NotImplementedError()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()

    def compute_rewards_to_go(self, buffer: EpisodeBuffer) -> FloatTensor:
        raise NotImplementedError()

    def compute_advantages(self, buffer: EpisodeBuffer) -> FloatTensor:
        raise NotImplementedError()

    def update_policy(self, buffer: EpisodeBuffer, advantages: FloatTensor):
        raise NotImplementedError()

    def update_value_function(self, buffer: EpisodeBuffer, rewards_to_go: FloatTensor):
        raise NotImplementedError()

    def policy_loss(
        self, buffer: EpisodeBuffer, advantages: FloatTensor
    ) -> FloatTensor:
        raise NotImplementedError()

    def entropy_loss(self, buffer: EpisodeBuffer) -> FloatTensor:
        raise NotImplementedError()

    def value_loss(
        self, buffer: EpisodeBuffer, rewards_to_go: FloatTensor
    ) -> FloatTensor:
        raise NotImplementedError()

    def vae_loss(self, buffer: EpisodeBuffer) -> FloatTensor:
        raise NotImplementedError()

    def update_from_batch(self, buffer: EpisodeBuffer, progress_bar: bool = False):
        """
        Pseudoe code steps for the inner loop:

        1) Compute Rewards to go
        2) Compute Advantage based on current value function
        3) Update the policy by maximizing the PPO loss

        4) Update the value function by minimizing the value loss"""
        # 1) Compute Rewards to go
        rewards_to_go = self.compute_rewards_to_go(buffer)

        # 2) Compute Advantage based on current value function
        advantages = self.compute_advantages(buffer)

        # 3) Update the policy by maximizing the PPO loss
        self.update_policy(buffer, advantages)

        # 4) Update the value function by minimizing the value loss
        self.update_value_function(buffer, rewards_to_go)

        raise NotImplementedError()

    def learn(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
        reset_buffer: bool = False,
        callback: BaseCallback | None = None,
    ):
        super().learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            reset_buffer=reset_buffer,
            callback=callback,
        )

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(model.state_inference.vae, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)

        # figure out what the accepatble args and kwargs are
        # Get the signature of the __init__ method of DiscretePPO
        sig = inspect.signature(cls.__init__)

        # Iterate through the parameters to separate args and kwargs
        args = []
        kwargs = []
        for param in sig.parameters.values():
            # Skip 'self' parameter for methods
            if param.name == "self":
                continue
            if param.default == inspect.Parameter.empty:
                args.append(param.name)
            else:
                kwargs.append(param.name)

        print(args, kwargs)

        state_inference_kwargs = {
            k: v
            for k, v in agent_config["state_inference_model"].items()
            if k in args + kwargs
        }
        print(state_inference_kwargs)
        return cls(task, vae, **state_inference_kwargs)

    def get_graph_laplacian(
        self, normalized: bool = True
    ) -> tuple[np.ndarray, Dict[Hashable, int]]:
        return self.transition_estimator.get_graph_laplacian(normalized=normalized)
