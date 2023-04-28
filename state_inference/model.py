from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from state_inference.pytorch_utils import DEVICE, gumbel_softmax, train


class ModelBase(nn.Module):
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
        _, z = self.encode(x)
        return torch.argmax(z, dim=-1)

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
        transition_model: nn.Module,
    ) -> None:
        super().__init__()

        self.observation_model = observation_model
        self.transition_model = transition_model

    def train_state_model(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        clip_grad: bool = None,
    ) -> List[torch.Tensor]:
        def preprocess_fn(
            batch_data: Tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
            return batch_data[0]

        return train(
            model=self.observation_model,
            train_loader=train_loader,
            optimizer=optimizer,
            clip_grad=clip_grad,
            preprocess=preprocess_fn,
        )

    def train_transition_model(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        clip_grad: bool = None,
    ):
        def _preprocess_fn(
            batch_data: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x, y = batch_data
            z_train = self.observation_model.encode_states(x)
            z_target = self.observation_model.encode_states(y)
            return z_train, z_target

        return train(
            model=self.observation_model,
            train_loader=train_loader,
            optimizer=optimizer,
            clip_grad=clip_grad,
            preprocess=_preprocess_fn,
        )

    def _smooth_state_estimates(self, x: torch.Tensor):
        """assumption is that x is a sequence of observations"""

        # encoded logits are (n_sequence * z_layers * z_dim)
        state_logits, z = self.observation_model.encode(x)

        # use z to predict transition logits

        # normalize (broadcasting)
        state_logits = state_logits - torch.logsumexp(state_logits, dim=2)[:, :, None]

        # get transition logits


class RewardModel:
    # Generative Model
    pass
