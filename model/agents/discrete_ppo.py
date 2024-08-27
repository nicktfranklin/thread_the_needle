import inspect
import logging
import os
from typing import Any, Dict, Hashable, Iterable, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback

# from stable_baselines3.common.policies import ActorCriticPolicy
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader

import model.state_inference.vae
from model.agents.utils.base_agent import BaseAgent
from model.state_inference.nets.mlp import MLP
from model.state_inference.vae import StateVae
from model.training.rollout_data import BaseBuffer, PpoBuffer
from task.utils import ActType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class PpoDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data["observations"])

    def collate_fn(batch):
        return {
            "observations": torch.stack([item["observations"] for item in batch]),
            "next_observations": torch.stack(
                [item["next_observations"] for item in batch]
            ),
            "rewards_to_go": torch.cat([item["rewards_to_go"] for item in batch]),
            "advantages": torch.cat([item["advantages"] for item in batch]),
            "actions": torch.cat([item["actions"] for item in batch]),
            "log_probs": torch.cat([item["log_probs"] for item in batch]),
        }


class DiscretePPO(BaseAgent, torch.nn.Module):
    minimum_episode_length = 2

    def __init__(
        self,
        env: gym.Env,
        state_inference_model: StateVae,
        gamma: float = 0.95,
        lr: float = 3e-4,
        n_steps: int = 2048,
        clip: float = 0.2,
        grad_clip: Optional[float] = 0.5,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        n_epochs: int = 10,
        batch_size: int = 64,
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
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.gamma = gamma
        self.clip = clip
        self.lr = lr
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size

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

        self.actor = MLP(
            input_size=self.embedding_size,
            hidden_sizes=[self.embedding_size],
            output_size=self.n_actions,
        )

        self.critic = MLP(
            input_size=self.embedding_size,
            hidden_sizes=[self.embedding_size],
            output_size=1,
        )
        self.optim = self._configure_optimizers(optim_kwargs)

    def _configure_optimizers(self, optim_kwargs: Optional[Dict[str, Any]] = None):
        optim_kwargs = optim_kwargs if optim_kwargs else dict()
        return torch.optim.AdamW(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.state_inference_model.parameters()),
            lr=self.lr,
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
        (logits, z), _ = self.embed(obs)

        logits = self.actor(z)
        dist = torch.distributions.Categorical(logits=logits)

        return dist.probs.detach().cpu().numpy()

    def get_state_hashkey(self, obs: Tensor) -> Hashable:
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def dehash_states(self, hashed_states: int | List[int]) -> torch.LongTensor:

        if isinstance(hashed_states, List):
            return torch.stack([self.dehash_states(h) for h in hashed_states])

        assert isinstance(hashed_states, (int, np.integer, torch.int))

        z = torch.zeros(self.state_inference_model.z_layers, dtype=torch.long)
        for ii in range(self.state_inference_model.z_layers - 1, -1, -1):
            z[ii] = hashed_states // self.hash_vector[ii]
            hashed_states = hashed_states % self.hash_vector[ii]
        return F.one_hot(z, self.state_inference_model.z_dim)

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
            - logits (Tensor): The logits tensor with shape (batch_size, z_layers, z_dim).
            - z (Tensor): The flattened latent representation tensor with shape (batch_size, latent_dim).

        """
        (logits, z), y_hat = self.state_inference_model(self._preprocess_obs(obs))
        z = self.state_inference_model.flatten_z(z)
        return (logits, z), y_hat

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

    def get_value(self, state_vec: FloatTensor) -> FloatTensor:
        return self.critic(state_vec)

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

        # get the embeddings
        (_, z), _ = self.embed(obs)

        # get the value function
        V = self.critic(z)

        # advantages = rewards to go - value function
        A = rtg.view(-1) - V.view(-1)

        # normalize the advantages for stability
        A = (A - A.mean()) / (A.std() + 1e-10)

        return A

    def make_dataset(self, buffer: PpoBuffer) -> dict[str, torch.Tensor]:
        observations_list = []
        next_observations_list = []
        rewards_to_go_list = []
        advantages_list = []
        actions_list = []
        log_probs_list = []

        for episode in buffer.iterator():
            episode_data = episode.get_dataset()
            episode_data = self.collocate(episode_data)

            # 1) Compute Rewards to go
            rewards_to_go = self.collocate(
                self.compute_rewards_to_go(episode_data["rewards"])
            )

            # 2) Compute Advantage based on current value function
            advantages = self.compute_advantages(
                episode_data["observations"], rewards_to_go
            )

            observations_list.append(episode_data["observations"])
            next_observations_list.append(episode_data["next_observations"])
            rewards_to_go_list.append(rewards_to_go)
            advantages_list.append(advantages)
            actions_list.append(episode_data["actions"])
            log_probs_list.append(episode_data["log_probs"])

        observations = torch.cat(observations_list, dim=0).cpu()
        next_observations = torch.cat(next_observations_list, dim=0).cpu()
        rewards_to_go = torch.cat(rewards_to_go_list, dim=0).cpu()
        advantages = torch.cat(advantages_list, dim=0).cpu()
        actions = torch.cat(actions_list, dim=0).cpu()
        log_probs = torch.cat(log_probs_list, dim=0).cpu()

        return PpoDataset(
            {
                "observations": observations,
                "next_observations": next_observations,
                "rewards_to_go": rewards_to_go,
                "advantages": advantages,
                "actions": actions,
                "log_probs": log_probs,
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

        datatset = self.make_dataset(buffer)
        n_cpus = os.cpu_count()
        dataloader = DataLoader(
            datatset, batch_size=self.batch_size, shuffle=True, num_workers=n_cpus
        )

        for _ in range(self.n_epochs):
            for batch in dataloader:
                batch = self.collocate(batch)
                observations = batch["observations"]
                next_observations = batch["next_observations"]
                rewards_to_go = batch["rewards_to_go"]
                advantages = batch["advantages"]
                actions = batch["actions"]
                log_probs = batch["log_probs"]

                # all loss computations go here!
                self.optim.zero_grad()

                # embeddings -- this is required for all the losses
                (logits, z), y_hat = self.embed(observations.float())

                # ELBO loss of the VAE
                # get the two components of the ELBO loss. Note, the VAE predicts the next oberservation
                try:
                    kl_loss = self.state_inference_model.kl_loss(logits)
                    recon_loss = self.state_inference_model.recontruction_loss(
                        self._preprocess_obs(next_observations),
                        y_hat,
                    )
                    vae_elbo = recon_loss + kl_loss
                except Exception as e:
                    print(logits)
                    raise e

                # ppo loss
                # get the policy logits (shape: batch_size x n_actions)
                action_logits = self.actor(z)
                dist = torch.distributions.Categorical(logits=action_logits)
                cur_log_probs = dist.log_prob(actions)
                old_log_probs = log_probs

                ratio = torch.exp(cur_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # value loss
                V = self.critic(z)
                value_loss = (rewards_to_go.view(-1).float() - V.view(-1)).pow(2).mean()

                # overall loss
                ppo_loss = actor_loss + value_loss
                loss = ppo_loss + vae_elbo
                loss.backward()

                if torch.isnan(loss):
                    print("Loss contains NaN values")
                    print("PPO Loss:", ppo_loss.item())
                    print("VAE ELBO Loss:", vae_elbo.item())
                    print("Actor Loss:", actor_loss.item())
                    print("Value Loss:", value_loss.item())
                    print("KL Loss:", kl_loss.item())
                    print("Reconstruction Loss:", recon_loss.item())
                    raise Exception("Loss contains NaN values")
                else:
                    self.optim.step()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optim.step()

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
                (_, z), _ = self.embed(torch.from_numpy(obs).float().to(self.device))

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
        reset_buffer: bool = False,
        capacity: Optional[int] = None,
        callback: MaybeCallback = None,
        buffer_kwargs: Dict[str, Any] | None = None,
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

        state_inference_kwargs = {
            k: v
            for k, v in agent_config["state_inference_model"].items()
            if k in args + kwargs
        }
        return cls(task, vae, **state_inference_kwargs)

    def get_state_values(self, state_key: Dict[int, int]) -> Dict[int, float]:

        z = self.dehash_states(list(state_key.keys()))
        z = self.collocate(z).float().flatten(start_dim=1)

        V = self.critic(z).detach().cpu().numpy()

        return {z0: v.item() for z0, v in zip(state_key.keys(), V)}
