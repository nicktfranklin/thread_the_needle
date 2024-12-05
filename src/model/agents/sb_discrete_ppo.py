import os
from logging import getLogger
from typing import ClassVar, Dict, Hashable, List, Type, TypeVar

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import tqdm
from gymnasium import spaces
from memory_profiler import profile
from stable_baselines3 import PPO as WrappedPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import FloatTensor

from .stable_baseline_clone.buffers import RolloutBuffer
from .stable_baseline_clone.policies import ActorCriticVaePolicy, BasePolicy
from .utils.base_agent import BaseAgent
from .utils.data import ViPpoDataset, ViPpoRolloutSample
from .utils.tabular_agent_pytorch import ModelBasedAgent

logger = getLogger(__name__)

SelfDPPO = TypeVar("SelfDPPO", bound="DiscretePpo")


class DiscretePpo(WrappedPPO, BaseAgent):
    """
    wrapper for PPO with useful functions
    """

    def __init__(self, *args, vae_coef: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_coef = vae_coef

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "CnnPolicy": ActorCriticVaePolicy,
    }
    rollout_buffer: RolloutBuffer
    policy: ActorCriticVaePolicy

    def get_pmf(self, obs: FloatTensor) -> FloatTensor:
        return (
            self.policy.get_distribution(
                preprocess_obs(obs.permute(0, 3, 1, 2).to(self.device), self.env.observation_space)
            )
            .distribution.probs.clone()
            .detach()
            .cpu()
            .numpy()
        )

    def get_value_fn(self, x: FloatTensor) -> FloatTensor:
        raise NotImplementedError

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError("Dict observation space is not supported")
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def eval(self):
        self.policy.set_training_mode(False)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # ~~~~~ Modified ~~~~~
            # save and log the successor observations
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                new_obs,
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                dones,
            )
            # ~~~~~~~~~~~~~~~~~~~~
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _anneal_vae(self) -> None:
        self.policy.anneal_vae_tau()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # anneal the vae tau
        self._anneal_vae()

        entropy_losses = []
        pg_losses, value_losses, vae_elbos = [], [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, vae_loss = self.policy.evaluate_actions(
                    rollout_data.observations, actions, rollout_data.next_observations
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                vae_elbo = vae_loss.kl_div + vae_loss.recon_loss

                # Logging
                pg_losses.append(policy_loss.item())
                vae_elbos.append(vae_elbo.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # # off policy value loss
                # off_policy_value_loss = F.mse_loss(
                #     rollout_data.off_policy_values, values_pred
                # )
                # off_policy_value_losses.append(off_policy_value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.vae_coef * vae_elbo
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/vae_elbo", np.mean(vae_elbos))
        self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record(
        #     "train/off_policy_value_loss", np.mean(off_policy_value_losses)
        # )
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def get_states(self, obs: FloatTensor, batch_size: int = 64) -> Hashable:
        is_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            states = self.policy.get_state_index(obs)
        if is_training:
            self.policy.train()
        return states

    def reset_state_indexer(self):
        self.policy.reset_state_indexer()

    def dehash_states(self, hashed_states: int | List[int]) -> torch.LongTensor:
        return self.policy.lookup_states(hashed_states)

    def get_state_values(self, state_key: Dict[int, int]) -> Dict[int, float]:

        z = self.dehash_states(list(state_key.keys()))
        z = z.float().flatten(start_dim=1)

        V = self.policy.predict_values(z).detach().cpu().numpy()

        return {z0: v.item() for z0, v in zip(state_key.keys(), V)}

    def learn(
        self: SelfDPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "DPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


class ViPPO(DiscretePpo):
    def __init__(self, *args, vi_coef: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.vi_coef = vi_coef

    def get_states(self, obs: FloatTensor) -> FloatTensor:
        is_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            states = self.policy.get_state_index(obs)
        if is_training:
            self.policy.train()
        return states

    def on_batch_end(self, batch_idx, epoch) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        self.policy.reset_state_indexer()

    def prepare_dataloader(self, rollout_buffer: RolloutBuffer) -> None:

        buffer_data = next(rollout_buffer.get())
        self.reset_state_indexer()

        states = self.get_states(buffer_data.observations)
        next_states = self.get_states(buffer_data.next_observations)

        n_states = torch.cat([states, next_states]).max().item() + 1
        self.logger.record("train/n_states", n_states)

        mdp = ModelBasedAgent(n_states, self.get_env().action_space.n, self.gamma)
        n = buffer_data.observations.shape[0]
        for ii in range(n):
            s, a, r, sp, done = (
                states[ii].item(),
                buffer_data.actions[ii].long().item(),
                buffer_data.rewards[ii].item(),
                next_states[ii].item(),
                bool(buffer_data.dones[ii].item()),
            )
            done = done if ii < n - 1 else True
            mdp.update(s, a, r, sp, done)

        value_function = mdp.estimate_value_function()

        vi_estimates = value_function[states].view(-1, 1)

        return torch.utils.data.DataLoader(
            ViPpoDataset(rollout_buffer, vi_estimates, device=self.device),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ViPpoDataset.collate_fn,
        )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # anneal the vae tau
        self._anneal_vae()

        entropy_losses = []
        pg_losses, value_losses, vae_elbos = [], [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Compute model based value estimates
            dataloader = self.prepare_dataloader(self.rollout_buffer)
            self.policy.train()

            # Do a complete pass on the rollout buffer
            all_states = set([])
            for batch, rollout_data in enumerate(dataloader):
                assert isinstance(rollout_data, ViPpoRolloutSample)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, vae_loss = self.policy.evaluate_actions(
                    rollout_data.observations.float(),
                    actions,
                    rollout_data.next_observations.float(),
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                vae_elbo = vae_loss.kl_div + vae_loss.recon_loss

                # Logging
                pg_losses.append(policy_loss.item())
                vae_elbos.append(vae_elbo.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # values_true = (
                #     (1 - self.vi_coef) * rollout_data.returns + self.vae_coef * rollout_data.vi_estimates
                # ).flatten()
                value_loss = F.mse_loss(rollout_data.vi_estimates, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.vae_coef * vae_elbo
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                self.on_batch_end(batch, epoch)
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            self.on_epoch_end(epoch)
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/vae_elbo", np.mean(vae_elbos))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class MemoryProfilingViPPO(ViPPO):
    def __init__(self, *args, log_freq=1, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_freq = log_freq
        self.verbose = verbose
        self.process = psutil.Process(os.getpid())

        # Storage for memory statistics
        self.cpu_mem_history: List[float] = []
        self.gpu_mem_history: Dict[int, List[float]] = {}
        self.batch_history: List[int] = []

        if torch.cuda.is_available():
            self.gpu_ids = list(range(torch.cuda.device_count()))
            for gpu_id in self.gpu_ids:
                self.gpu_mem_history[gpu_id] = []

        elif torch.mps.is_available():
            self.gpu_ids = list(range(torch.mps.device_count()))
            for gpu_id in self.gpu_ids:
                self.gpu_mem_history[gpu_id] = []
        else:
            self.gpu_ids = []

        if verbose:
            print("Memory Profiling")
            print(f"Logging frequency: {log_freq} batches")
            print(f"Available GPUs: {self.gpu_ids}")

    def _get_gpu_memory_usage(self, gpu_id: int) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
        elif torch.mps.is_available():
            return torch.mps.current_allocated_memory() / 1024 / 1024
        return 0.0

    def _get_cpu_memory_usage(self) -> float:
        """Get CPU memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _log_memory_usage(self, batch_idx: int, epoch: int):
        """Log memory usage to console and/or TensorBoard."""
        cpu_mem = self._get_cpu_memory_usage()
        self.cpu_mem_history.append(cpu_mem)

        # self.logger.record("Memory/CPU_MB", cpu_mem)

        for gpu_id in self.gpu_ids:
            gpu_mem = self._get_gpu_memory_usage(gpu_id)
            self.gpu_mem_history[gpu_id].append(gpu_mem)

            # self.logger.record(f"Memory/GPU_{gpu_id}_MB", gpu_mem)

    def on_batch_end(self, batch_idx, epoch) -> None:
        """Called at the end of each batch."""
        if batch_idx % self.log_freq == 0:
            self._log_memory_usage(batch_idx, epoch)
            self.batch_history.append(batch_idx)

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        # Calculate and log memory statistics for the epoch
        cpu_mean = np.mean(self.cpu_mem_history[-self.log_freq :])
        cpu_peak = np.max(self.cpu_mem_history[-self.log_freq :])

        self.logger.record("Memory/CPU_Mean_MB", cpu_mean)
        self.logger.record("Memory/CPU_Peak_MB", cpu_peak)

        stats = [
            f"Epoch {epoch} Summary:",
            f"CPU Memory - Mean: {cpu_mean:.1f}MB, Peak: {cpu_peak:.1f}MB",
        ]

        for gpu_id in self.gpu_ids:
            gpu_mean = np.mean(self.gpu_mem_history[gpu_id][-self.log_freq :])
            gpu_peak = np.max(self.gpu_mem_history[gpu_id][-self.log_freq :])

            logger.info(f"GPU{gpu_id} Memory - Mean: {gpu_mean:.1f}MB, Peak: {gpu_peak:.1f}MB")

            self.logger.record(f"Memory/GPU_{gpu_id}_Mean_MB", gpu_mean)
            self.logger.record(f"Memory/GPU_{gpu_id}_Peak_MB", gpu_peak)

            stats.append(f"GPU{gpu_id} Memory - Mean: {gpu_mean:.1f}MB, Peak: {gpu_peak:.1f}MB")

        if self.verbose:
            print("\n".join(stats))
