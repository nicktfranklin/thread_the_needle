from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from state_inference.model.vae import MLP, ModelBase, StateVae
from state_inference.utils.pytorch_utils import DEVICE


class TransitionPredictor(MLP):
    """a model that states in states and predicts a distribution over states"""

    def __init__(
        self,
        hidden_sizes: List[int],
        z_dim: int,
        z_layers: int,
        dropout: float = 0.01,
    ):
        super().__init__(z_dim * z_layers, hidden_sizes, z_dim * z_layers, dropout)
        self.z_dim = z_dim
        self.z_layers = z_layers

        self.net.pop(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            super().forward(x.view(x.shape[0], -1)).view(-1, self.z_layers, self.z_dim)
        )

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch  # unpack the batch
        inputs, targets = inputs.to(DEVICE).float(), inputs.to(DEVICE).float()

        outs = self(inputs)  # apply the model
        loss = F.cross_entropy(outs, targets)  # compute the (cross entropy) loss
        return loss


class PomdpModel(ModelBase):
    def __init__(
        self,
        observation_model: StateVae,
        forward_transitions: nn.Module,
        reverse_transitions: nn.Module,
    ) -> None:
        super().__init__()

        self.observation_model = observation_model
        self.forward_transitions = forward_transitions
        self.reverse_transitions = reverse_transitions

    @staticmethod
    def combine_log_probs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Takes in two logit tensors of size (n X z_layers x z_dim)
        and outputs the normalized log probability vectors
        """
        x = x - torch.logsumexp(x, dim=2)[:, :, None]
        y = y - torch.logsumexp(y, dim=2)[:, :, None]

        z = x + y
        return z - torch.logsumexp(z, dim=2)[:, :, None]

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch  # unpack the batch
        inputs, targets = inputs.to(DEVICE).float(), inputs.to(DEVICE).float()

        # encode
        _, inputs_z = self.observation_model.encode(inputs)
        _, targets_z = self.observation_model.encode(targets)

        # predict the transition
        outs_forward = self.forward_transitions(inputs_z.float())
        loss_forward = F.cross_entropy(outs_forward, targets_z.float())

        # predict the inverse transition
        out_reverse = self.reverse_transitions(targets_z.float())
        loss_reverse = F.cross_entropy(out_reverse, inputs_z.float())

        return loss_forward + loss_reverse

    def forward(self, x):
        raise NotImplementedError


class RewardModel:
    # Generative Model
    pass
