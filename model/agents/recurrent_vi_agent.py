from random import choice
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from model.agents.constants import (
    ALPHA,
    BATCH_LENGTH,
    BATCH_SIZE,
    EPSILON,
    GAMMA,
    GRAD_CLIP,
    MAX_SEQUENCE_LEN,
    N_EPOCHS,
    N_ITER_VALUE_ITERATION,
    SOFTMAX_GAIN,
)
from model.agents.value_iteration import ValueIterationAgent
from model.data import OaroTuple
from model.state_inference.vae import RecurrentVae
from utils.data import RecurrentVaeDataset, TransitionVaeDataset
from utils.pytorch_utils import DEVICE, convert_8bit_to_float, maybe_convert_to_tensor


class RecurrentViAgent(ValueIterationAgent):
    def __init__(
        self,
        task,
        state_inference_model: RecurrentVae,
        set_action: Set[ActType],
        optim_kwargs: Dict[str, Any] | None = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
        epsilon: float = EPSILON,
        batch_length: int = BATCH_LENGTH,
        n_epochs: int = N_EPOCHS,
        alpha: float = ALPHA,
        max_sequence_len: int = MAX_SEQUENCE_LEN,
    ) -> None:
        super().__init__(
            task,
            state_inference_model,
            set_action,
            optim_kwargs,
            grad_clip,
            batch_size,
            gamma,
            n_iter,
            softmax_gain,
            epsilon,
            batch_length,
            n_epochs,
            alpha,
        )
        self.max_sequence_len = max_sequence_len

    @staticmethod
    def construct_dataloader_from_obs(
        batch_size: int, list_obs: List[Tensor], max_seq_len: int = MAX_SEQUENCE_LEN
    ) -> DataLoader:
        obs = torch.stack([obs.obs for obs in list_obs])
        actions = F.one_hot(torch.tensor([obs.a for obs in list_obs]))
        index = [obs.index for obs in list_obs]
        return RecurrentVaeDataset.construct_dataloader(
            obs, actions, index, max_seq_len, batch_size
        )

    def _prep_vae_dataloader(
        self,
        batch_size: int,
    ):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
        """
        return RecurrentViAgent.construct_dataloader_from_obs(
            batch_size, self.rollout_buffer.get_all(), self.max_sequence_len
        )

    def contruct_validation_dataloader(self, sample_size, seq_len):
        # assert (
        #     sample_size % seq_len == 0
        # ), "Sample size must be an interger multiple of sequence length"
        validation_obs = []
        for t in range(sample_size // seq_len):
            obs_prev = self.task.reset()[0]

            for _ in range(seq_len):
                action = choice(list(self.action_space))
                obs, rew, terminated, _, _ = self.task.step(action)
                obs_tuple = OaroTuple(
                    obs=torch.tensor(obs_prev),
                    a=action,
                    r=rew,
                    obsp=torch.tensor(obs),
                    index=t,
                )
                validation_obs.append(obs_tuple)

                if terminated:
                    break
        return RecurrentViAgent.construct_dataloader_from_obs(
            batch_size=len(validation_obs), list_obs=validation_obs, max_seq_len=seq_len
        )

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _init_state(self) -> Tensor:
        # unbatch state for the agent (not vae training)
        return torch.zeros(
            self.state_inference_model.z_dim * self.state_inference_model.z_layers
        )

    def _get_hashed_state(self, obs: Tensor, state_prev: Optional[Tensor]):
        obs = maybe_convert_to_tensor(obs)
        obs = convert_8bit_to_float(obs)
        with torch.no_grad():
            state = self.state_inference_model.get_state(obs, state_prev)
        return state.dot(self.hash_vector)

    def _update_hidden_state(self, obs: Tensor, state: Tensor) -> Tensor:
        return self.state_inference_model.update_hidden_state(obs, state)

    def predict(
        self,
        obs: Tensor,
        state: Tensor,
        episode_start=None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        obs = maybe_convert_to_tensor(obs)
        obs = convert_8bit_to_float(obs)

        # e-greedy sampling
        if not deterministic and np.random.rand() < self.policy.epsilon:
            a = torch.tensor([np.random.randint(self.policy.n_actions)])
        else:
            hashed_sucessor_state = self._get_hashed_state(obs, state)

            p = self.policy.get_distribution(hashed_sucessor_state)

            # sample the action
            a = p.get_actions(deterministic=deterministic)

        # update the RNN Hidden state
        next_state = self._update_hidden_state(obs, state)

        return a, next_state

    def _update_rollout_policy(
        self, obs: OaroTuple, state: None, state_prev: None
    ) -> None:
        # prep the observations for the RNN
        o = convert_8bit_to_float(obs.obs[None, ...]).to(DEVICE)
        op = convert_8bit_to_float(obs.obsp[None, ...]).to(DEVICE)

        state = state.view(1, -1).to(DEVICE)
        state_prev = state_prev.view(1, -1).to(DEVICE)

        # pass through the RNN to get the embedding
        _, z = self.state_inference_model.encode_from_state(o, state)
        _, zp = self.state_inference_model.encode_from_state(op, state_prev)

        # convert from one hot tensor to int array
        s = torch.argmax(z, dim=-1).detach().cpu().view(-1).numpy()
        sp = torch.argmax(zp, dim=-1).detach().cpu().view(-1).numpy()

        s = s.dot(self.hash_vector)
        sp = sp.dot(self.hash_vector)
        self.update_qvalues(s, obs.a, obs.r, sp)


class ViControlableStateInf(ValueIterationAgent):
    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        obs = self.rollout_buffer.get_tensor("obs")
        obsp = self.rollout_buffer.get_tensor("obsp")
        a = F.one_hot(self.rollout_buffer.get_tensor("a"), num_classes=4).float()

        dataset = TransitionVaeDataset(obs, a, obsp)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
