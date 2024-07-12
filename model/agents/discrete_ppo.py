import inspect
import logging
from typing import Any, Dict, Hashable, Iterable, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback

# from stable_baselines3.common.policies import ActorCriticPolicy
from torch import FloatTensor, Tensor

import model.state_inference.vae
from model.agents.utils.base_agent import BaseAgent
from model.state_inference.nets.mlp import MLP
from model.state_inference.vae import StateVae
from model.training.rollout_data import Episode, EpisodeBuffer, PpoBuffer
from task.utils import ActType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class DiscretePPO(BaseAgent):

    def __init__(
        self,
        env: gym.Env,
        state_inference_model: StateVae,
        gamma: float = 0.95,
        lr: float = 0.005,
        clip: float = 0.2,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(env)
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.optim = self._configure_optimizers(optim_kwargs)
        self.gamma = gamma

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

        self.embedding_size = (
            self.state_inference_model.z_dim * self.state_inference_model.z_layers
        )
        self.n_actions = env.action_space.n

        self.policy = MLP(
            input_size=self.embedding_size,
            hidden_sizes=[self.embedding_size],
            output_size=self.n_actions,
        )

        self.critic = MLP(
            input_size=self.embedding_size,
            hidden_sizes=[self.embedding_size],
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

    def embed(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        return self.state_inference_model(self._preprocess_obs(obs))

    def get_action(self, state_vec: FloatTensor) -> tuple[int, FloatTensor]:
        logits = self.policy(state_vec)

        dist = torch.distributions.Categorical(logits=logits)

        # Sample (detached from computation graph)
        action = dist.sample()

        # Compute log probability (connected to computation graph)
        log_probs = dist.log_prob(action)

        return action.item(), log_probs

    def get_value(self, state_vec: FloatTensor) -> FloatTensor:
        return self.critic(state_vec)

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()

    def compute_rewards_to_go(self, rewards: np.array) -> FloatTensor:
        """Compute the rewards to go for a given episode via recursion."""

        # base case 1
        if len(rewards) == 0:
            return []

        # base case 2
        rtg = self.compute_rewards_to_go(rewards[1:])
        if len(rtg) == 0:
            return [rewards[0]]

        # recursive case
        return [rewards[0] + self.gamma * rtg[0]] + rtg

    @torch.no_grad()
    def compute_advantages(self, obs: FloatTensor, rtg: FloatTensor) -> FloatTensor:
        """Compute the advantages for a given episode. Does not require gradients."""

        # get the embeddings
        (_, z), _ = self.embed(obs)

        # get the value function
        V = self.critic(z)

        # advantages = rewards to go - value function
        A = rtg - V

        # normalize the advantages for stability
        A = (A - A.mean()) / (A.std() + 1e-10)

        return A

    def _configure_optimizers(self, **kwargs):
        raise NotImplementedError()

    def update_from_batch(self, buffer: PpoBuffer, progress_bar: bool = False):
        """
        Pseudo code steps for the inner loop:

        1) Compute Rewards to go
        2) Compute Advantage based on current value function
        3) Update the policy by maximizing the PPO loss
        4) Update the value function by minimizing the value loss"""

        self.train()

        for episode in buffer.iterator:

            episode_data = episode.get_dataset()

            # 1) Compute Rewards to go
            rewards_to_go = self.compute_rewards_to_go(episode_data["rewards"])

            # 2) Compute Advantage based on current value function
            advantages = self.compute_advantages(
                episode_data["observations"], rewards_to_go
            )

            for _ in range(self.iterations_per_batch):
                # all loss computations go here!
                pass

            # 3) Update the policy by maximizing the PPO loss
            # self.update_policy(episode, advantages)

            # 4) Update the value function by minimizing the value loss
            # self.update_value_function(episode)

            # 5) Update the state inference model

        raise NotImplementedError()

    def collect_rollouts(
        self,
        n_rollout_steps: int,
        rollout_buffer: PpoBuffer,
        callback: BaseCallback,
        progress_bar: Iterable | bool = False,
    ) -> bool:
        task = self.get_task()
        obs = task.reset()[0]

        callback.on_rollout_start()

        for _ in range(n_rollout_steps):
            self.num_timesteps += 1

            # do not collect the gradient for the rollout
            with torch.no_grad():
                # vae encode
                (embedding_logits, z), y_hat = self.embed(obs)

                # policy
                action, log_probs = self.get_action(z)

            outcome_tuple = task.step(action)

            rollout_buffer.add(
                obs,
                action,
                outcome_tuple,
                log_probs,
            )

            # get the next obs from the observation tuple
            obs, reward, done, truncated, _ = outcome_tuple

            # calculate the advantages

            if done or truncated:
                obs = task.reset()[0]

            callback.update_locals(locals())
            if not callback.on_step():
                return False

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def learn(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
        reset_buffer: bool = False,
        capacity: Optional[int] = None,
        callback: MaybeCallback = None,
        buffer_class: str | None = None,
        buffer_kwargs: Dict[str, Any] | None = None,
        **kwargs,
    ):

        buffer_kwargs = buffer_kwargs if buffer_kwargs else dict()
        self.rollout_buffer = PpoBuffer(capacity=capacity, **buffer_kwargs)

        callback = self._init_callback(callback, progress_bar=progress_bar)
        callback.on_training_start(locals(), globals())

        # alternate between collecting rollouts and batch updates
        n_rollout_steps = self.n_steps if self.n_steps is not None else total_timesteps

        self.num_timesteps = 0
        while self.num_timesteps < total_timesteps:
            if reset_buffer:
                self.rollout_buffer.reset_buffer()

            n_rollout_steps = min(n_rollout_steps, total_timesteps - self.num_timesteps)

            if not self.collect_rollouts(
                n_rollout_steps,
                self.rollout_buffer,
                callback=callback,
                progress_bar=progress_bar,
            ):
                break

            self.update_from_batch(self.rollout_buffer, progress_bar=False)

        callback.on_training_end()

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
