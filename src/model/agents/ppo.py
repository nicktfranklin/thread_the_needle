import datetime
import inspect
import os
from typing import Any, Dict, Hashable, Iterable, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO as WrappedPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback

# from stable_baselines3.common.policies import ActorCriticPolicy
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.model.state_inference.vae
from src.task.utils import ActType
from src.utils.pytorch_utils import DEVICE, convert_8bit_to_float

from ..state_inference.nets.cnn import CnnEncoder
from ..training.rollout_data import PpoBuffer, RolloutBuffer
from .utils.base_agent import BaseAgent
from .utils.data import PpoDataset


class StableBaselinesPPO(WrappedPPO, BaseAgent):
    """
    wrapper for PPO with useful functions
    """

    def update_from_batch(self, batch: RolloutBuffer):
        raise NotImplementedError

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:
        return (
            self.policy.get_distribution(obs.permute(0, 3, 1, 2)).distribution.probs.clone().detach().numpy()
        )

    def get_states(self, obs: Tensor) -> Hashable:
        raise NotImplementedError

    def get_value_fn(self, obs: FloatTensor) -> FloatTensor:
        return self.predict(obs)[1].detach().cpu().numpy()


class PPO(BaseAgent, torch.nn.Module):
    minimum_episode_length = 2

    def __init__(
        self,
        env: gym.Env,
        feature_extractor: CnnEncoder,
        gamma: float = 0.95,
        n_steps: int = 2048,
        clip: float = 0.2,
        grad_clip: Optional[float] = 0.5,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        n_epochs: int = 10,
        batch_size: int = 64,
        epsilon: float = 1e-3,  # for numerical stability
        ppo_loss_weight: float = 1.0,  # for balancing the PPO loss and the VAE loss
        entropy_weight: float = 1.0,  # for balancing the entropy loss and the value loss
    ) -> None:
        """
        Initializes the DiscretePPO agent.

        Args:
            env (gym.Env): The environment.
            state_inference_model (StateVae): The state inference model.
            gamma (float): Discount factor for future rewards. Defaults to 0.95.
            lr (float): Learning rate for the optimizer. Defaults to 3e-4.
            clip (float): Clipping parameter for the PPO loss. Defaults to 0.2.
            grad_clip (float, optional): Gradient clipping parameter. Defaults to 0.5.
            optim_kwargs (Dict[str, Any], optional): Additional keyword arguments for the optimizer. Defaults to None.
            n_epochs (int, optional): Number of epochs for training per rollout batch. Defaults to 10.

        Todo:
            - Add more detailed description for each parameter.
        """
        super().__init__(env)
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.clip = clip
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.ppo_loss_weight = ppo_loss_weight
        self.entropy_weight = entropy_weight

        self.embedding_size = self.feature_extractor.embedding_dim
        self.n_actions = env.action_space.n

        self.actor_net = nn.Linear(self.embedding_size, self.n_actions)
        self.critic = nn.Linear(self.embedding_size, 1)
        self.optim = self._configure_optimizers(optim_kwargs)

    def _configure_optimizers(self, optim_kwargs: Optional[Dict[str, Any]] = None):
        optim_kwargs = optim_kwargs if optim_kwargs else dict()
        if not hasattr(optim_kwargs, "lr"):
            optim_kwargs["lr"] = 3e-4
        return torch.optim.AdamW(
            list(self.actor_net.parameters())
            + list(self.critic.parameters())
            + list(self.feature_extractor.parameters()),
            **optim_kwargs,
        )

    def _init_state(self):
        return None

    @property
    def device(self):
        return next(self.parameters()).device

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(self.collocate(obs))
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    @torch.no_grad()
    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        z = self.embed(obs)

        logits = self.actor(z.float())
        dist = torch.distributions.Categorical(logits=logits)

        return dist.probs.detach().cpu().numpy()

    @torch.no_grad()
    def get_value_fn(self, obs: FloatTensor) -> FloatTensor:
        z = self.embed(obs.float())
        return self.critic(z.float()).detach().cpu().numpy()

    def actor(self, x: FloatTensor) -> FloatTensor:
        """we use e-softmax policy here, mostly for numerical stability during training"""

        logits = self.actor_net(x)

        # Step 1: Normalize the logits using log-softmax
        log_normalized_probs = F.log_softmax(logits * self.entropy_weight, dim=-1)

        return log_normalized_probs

    def get_states(self, obs: Tensor) -> Hashable:
        raise NotImplementedError()

    def dehash_states(self, hashed_states: int | List[int]) -> torch.LongTensor:
        raise NotImplementedError()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()

    def save(self, model_path: str):
        """
        Save the model parameters to a file.

        :param model_path: The path to save the model.
        """
        torch.save(self.state_dict(), model_path)

    def embed(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Embeds the given observation into a latent space representation.

        Args:
            obs (Tensor): The input observation tensor. It should have the shape (batch_size, H, W, C).

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the logits tensor and the flattened latent representation tensor.
            - z (Tensor): The flattened latent representation tensor with shape (batch_size, latent_dim).

        """

        return self.feature_extractor(self._preprocess_obs(obs))

    def get_action(self, state_vec: FloatTensor) -> tuple[int, FloatTensor]:
        """returns action and log probability of the action. Log probability is needed for the PPO loss"""
        assert isinstance(state_vec, torch.Tensor)
        assert state_vec.dtype == torch.float32

        logits = self.actor(state_vec)

        dist = torch.distributions.Categorical(logits=logits)

        # Sample (detached from computation graph)
        action = dist.sample()

        # Compute log probability (connected to computation graph)
        log_probs = dist.log_prob(action)

        return action.item(), log_probs

    def compute_rewards_to_go(self, rewards: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute the rewards to go for a given episode via recursion."""

        # base case 1
        if len(rewards) == 0:
            return torch.tensor([])

        # base case 2
        rtg = self.compute_rewards_to_go(rewards[1:])
        if len(rtg) == 0:
            return torch.tensor([rewards[0]])

        # recursive case
        return torch.cat([torch.tensor([rewards[0] + self.gamma * rtg[0]]), rtg])

    @torch.no_grad()
    def compute_advantages(self, obs: FloatTensor, rtg: FloatTensor) -> FloatTensor:
        """Compute the advantages for a given episode. Does not require gradients."""
        assert isinstance(obs, torch.Tensor)
        assert isinstance(rtg, torch.Tensor)
        self.eval()

        # Ensure the inputs are of type float32
        obs = obs.float()
        rtg = rtg.float()

        # get the embeddings
        z = self.embed(obs)

        # get the value function
        V = self.critic(z.float())

        # advantages = rewards to go - value function
        A = rtg.view(-1) - V.view(-1).detach()

        # normalize the advantages for stability
        if A.shape[0] > 1:
            A = (A - A.mean()) / (A.std() + 1e-10)

        return A

    def make_dataset(self, buffer: PpoBuffer) -> dict[str, torch.Tensor]:
        observations_list = []
        next_observations_list = []
        rewards_to_go_list = []
        advantages_list = []
        actions_list = []
        log_probs_list = []
        episode_reward = []

        for episode in buffer.iterator():
            episode_data = episode.get_dataset()
            episode_data = self.collocate(episode_data)

            # 1) Compute Rewards to go
            rewards_to_go = self.collocate(self.compute_rewards_to_go(episode_data["rewards"]))
            # print(rewards_to_go)
            # print(episode_data["rewards"])
            # raise Exception("Stop here")

            # 2) Compute Advantage based on current value function
            advantages = self.compute_advantages(episode_data["observations"], rewards_to_go)

            observations_list.append(episode_data["observations"])
            next_observations_list.append(episode_data["next_observations"])
            rewards_to_go_list.append(rewards_to_go)
            advantages_list.append(advantages)
            actions_list.append(episode_data["actions"])
            log_probs_list.append(episode_data["log_probs"])
            episode_reward.append(episode_data["rewards"])

        observations = torch.cat(observations_list, dim=0).cpu()
        next_observations = torch.cat(next_observations_list, dim=0).cpu()
        rewards_to_go = torch.cat(rewards_to_go_list, dim=0).cpu()
        advantages = torch.cat(advantages_list, dim=0).cpu()
        actions = torch.cat(actions_list, dim=0).cpu()
        log_probs = torch.cat(log_probs_list, dim=0).cpu()
        episode_reward = torch.cat(episode_reward, dim=0).cpu()

        return PpoDataset(
            {
                "observations": observations,
                "next_observations": next_observations,
                "rewards_to_go": rewards_to_go,
                "advantages": advantages,
                "actions": actions,
                "log_probs": log_probs,
                "episode_reward": episode_reward,
            }
        )

    def update_from_batch(
        self,
        buffer: PpoBuffer,
        progress_bar: bool = False,
    ):
        """
        Pseudo code steps for the inner loop:

        1) Compute Rewards to go
        2) Compute Advantage based on current value function
        3) Update the policy by maximizing the PPO loss
        4) Update the value function by minimizing the value loss"""

        self.train()

        for _ in range(self.n_epochs):

            datatset = self.make_dataset(buffer)
            dataloader = DataLoader(datatset, batch_size=self.batch_size, shuffle=True)

            batch_reward = 0
            for batch in dataloader:
                batch = self.collocate(batch)
                observations = batch["observations"]
                rewards_to_go = batch["rewards_to_go"]
                advantages = batch["advantages"]
                actions = batch["actions"]
                log_probs = batch["log_probs"]

                batch_reward += batch["episode_reward"].sum().item()

                # all loss computations go here!
                self.optim.zero_grad()

                # embeddings -- this is required for all the losses
                z = self.embed(observations)
                z = z.float()

                # critic
                V = self.critic(z)
                value_loss = (rewards_to_go.view(-1) - V.view(-1)).pow(2).mean()

                # advantages = rewards_to_go - V.detach()

                # ppo loss
                # get the policy logits (shape: batch_size x n_actions)
                actor_log_probs = self.actor(z)
                dist = torch.distributions.Categorical(logits=actor_log_probs)
                cur_log_probs = dist.log_prob(actions.long())  # actions should be long for Categorical
                old_log_probs = log_probs

                ratio = torch.exp(cur_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # overall loss
                loss = actor_loss + value_loss
                loss.backward()

                if self.log_tensorboard:
                    self.writer.add_scalar("Loss/Total", loss.item(), self.episode_number)
                    self.writer.add_scalar("Loss/PPO", loss.item(), self.episode_number)

                    self.episode_number += 1

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            if self.log_tensorboard:
                self.writer.add_scalar("Episode/Reward", batch_reward, self.batch_number)
                self.batch_number += 1

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

        self.eval()
        for _ in range(n_rollout_steps):
            self.num_timesteps += 1

            # do not collect the gradient for the rollout
            with torch.no_grad():
                # vae encode
                z = self.embed(torch.from_numpy(obs).float().to(self.device))

                # policy
                action, log_probs = self.get_action(z.float())

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
        reset_buffer: bool = True,
        capacity: Optional[int] = None,
        callback: MaybeCallback = None,
        buffer_kwargs: Dict[str, Any] | None = None,
        log_tensorboard: bool = True,
        tensoboard_log_tag: str | None = None,
        **kwargs,
    ):
        buffer_kwargs = buffer_kwargs if buffer_kwargs else dict()
        capacity = capacity if capacity else self.n_steps
        self.rollout_buffer = PpoBuffer(capacity=capacity, **buffer_kwargs)

        callback = self._init_callback(callback, progress_bar=progress_bar)
        callback.on_training_start(locals(), globals())
        # alternate between collecting rollouts and batch updates
        n_rollout_steps = self.n_steps if self.n_steps is not None else total_timesteps

        self.num_timesteps = 0
        if log_tensorboard:
            self.batch_number = 0
            self.episode_number = 0
            current_date = datetime.date.today().strftime("%b%d_%H-%M")
            log_subdir = f"{current_date}_{tensoboard_log_tag}" if tensoboard_log_tag else current_date
            log_dir = os.path.join("runs", log_subdir)
            self.writer = SummaryWriter(log_dir=log_dir)
        self.log_tensorboard = log_tensorboard
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

            if reset_buffer:
                self.rollout_buffer.finish()

            self.update_from_batch(self.rollout_buffer, progress_bar=False)

        callback.on_training_end()
        self.writer.flush()

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        input_shape = (1, env_kwargs["map_height"], env_kwargs["map_height"])
        FeatureExtractor = getattr(src.model.state_inference.nets, agent_config["feature_extractor"]["class"])
        feature_extractor = FeatureExtractor(
            input_shape=input_shape, **agent_config["feature_extractor"]["kwargs"]
        )

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

        return cls(
            task,
            feature_extractor,
            **agent_config["optimizer_kwargs"],
        )

    def get_state_values(self, state_key: Dict[int, int]) -> Dict[int, float]:

        z = self.dehash_states(list(state_key.keys()))
        z = self.collocate(z).float().flatten(start_dim=1)

        V = self.critic(z).detach().cpu().numpy()

        return {z0: v.item() for z0, v in zip(state_key.keys(), V)}
