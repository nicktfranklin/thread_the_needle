from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from state_inference.utils.pytorch_utils import DEVICE, gumbel_softmax


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

            _, z = self.encode(x.to(DEVICE))

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


class DecoderWithActions(ModelBase):
    def __init__(
        self,
        embedding_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        ouput_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.mlp = Decoder(embedding_size, hidden_sizes, ouput_size, dropout)
        self.latent_embedding = nn.Linear(embedding_size, embedding_size)
        self.action_embedding = nn.Linear(n_actions, embedding_size)

    def forward(self, latents, action):
        x = self.latent_embedding(latents) + self.action_embedding(action)
        x = F.relu(x)
        x = self.mlp(x)
        return x

    def loss(self, latents, actions, targets):
        y_hat = self(latents, actions)
        # return y_hat
        return F.mse_loss(y_hat, torch.flatten(targets, start_dim=1))


class TransitionStateVae(StateVae):
    def decode(self, z, action):
        z_reshaped = z.view(-1, self.z_layers * self.z_dim).float()
        return self.decoder(z_reshaped.float(), action.float())

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def loss(self, batch_data=List[torch.Tensor]) -> torch.Tensor:
        obs, actions, obsp = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()
        obsp = obsp.to(DEVICE).float()

        logits, z = self.encode(obs)

        # # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(
            z.view(-1, self.z_layers * self.z_dim).float(), actions, obsp
        )

        return recon_loss + kl_loss * self.beta
