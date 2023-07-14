from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from state_inference.utils.pytorch_utils import DEVICE, gumbel_softmax


class ModelBase(nn.Module):
    device = DEVICE

    def __init__(self):
        super().__init__()

    def configure_optimizers(self, lr: float = 3e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def forward(self, x):
        raise NotImplementedError

    def prep_next_batch(self):
        pass


class MLP(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.nin = input_size
        self.nout = output_size

        # define a simple MLP neural net
        self.net = []
        hidden_size = [self.nin] + hidden_sizes + [self.nout]
        for h0, h1 in zip(hidden_size, hidden_size[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.BatchNorm1d(h1),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                ]
            )

        # pop the last ReLU and dropout layers for the output
        self.net.pop()
        self.net.pop()

        self.net = nn.Sequential(*self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.view(x.shape[0], -1))


class Decoder(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__(input_size, hidden_sizes, output_size, dropout)
        self.net.pop(-1)
        self.net.append(torch.nn.Sigmoid())


class StateVae(ModelBase):
    def __init__(
        self,
        encoder: ModelBase,
        decoder: ModelBase,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        """
        Note: larger values of beta result in more independent state values
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_layers = z_layers
        self.z_dim = z_dim
        self.beta = beta
        self.tau = tau
        self.gamma = gamma

    def reparameterize(self, logits):
        # either sample the state or take the argmax
        if self.training:
            z = gumbel_softmax(logits=logits, tau=self.tau, hard=False)
        else:
            s = torch.argmax(logits, dim=-1)  # tensor of n_batch * self.z_n_layers
            z = F.one_hot(s, num_classes=self.z_dim)
        return z

    def encode(self, x):
        # reshape encoder output to (n_batch, z_layers, z_dim)
        logits = self.encoder(x).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def decode(self, z):
        return self.decoder(z.view(-1, self.z_layers * self.z_dim).float())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, z = self.encode(x)
        return (logits, z), self.decode(z).view(x.shape)  # preserve original shape

    def kl_loss(self, logits):
        return Categorical(logits=logits).entropy().mean()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE).float()
        (logits, _), x_hat = self(x)

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = F.mse_loss(x_hat, x)

        return recon_loss + kl_loss * self.beta

    def get_state(self, x):
        self.eval()
        with torch.no_grad():
            # expand if unbatched
            assert x.view(-1).shape[0] % self.encoder.nin == 0
            if x.view(-1).shape[0] == self.encoder.nin:
                x = x[None, ...]

            _, z = self.encode(x.to(self.device))

            state_vars = torch.argmax(z, dim=-1).detach().cpu().numpy()
        return state_vars

    def decode_state(self, s: Tuple[int]):
        self.eval()
        z = (
            F.one_hot(torch.Tensor(s).to(torch.int64).to(DEVICE), self.z_dim)
            .view(-1)
            .unsqueeze(0)
        )
        with torch.no_grad():
            return self.decode(z).detach().cpu().numpy()

    def anneal_tau(self):
        self.tau *= self.gamma

    def prep_next_batch(self):
        self.anneal_tau()


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
