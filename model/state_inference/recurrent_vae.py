from typing import List, Optional, Tuple

import torch
import torch.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from model.state_inference.constants import INPUT_SHAPE, VAE_TAU_ANNEALING_RATE
from model.state_inference.vae import StateVae
from utils.pytorch_utils import DEVICE, assert_correct_end_shape


class LstmVae(StateVae):
    """A recurrent version of StateVae, where LstmVae reduces to
    StateVAE when the weights of the LSTM are zero."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        tau_annealing_rate: float = VAE_TAU_ANNEALING_RATE,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        tau_is_parameter: bool = False,
    ):
        super().__init__(
            encoder,
            decoder,
            z_dim,
            z_layers,
            beta,
            tau,
            tau_annealing_rate,
            input_shape,
            tau_is_parameter,
        )
        self.lstm = nn.LSTM(
            hidden_size=z_dim * z_layers, input_size=z_dim * z_layers, batch_first=False
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The encoder takes in a sequence of observations and returns
        an encoding of each of the observations in the sequence.  The logic is a
        filtering operation.

        Should handle batched or unbatched inputs, but should always assume a sequence.

        input: x (L, B, C, H, W) or (L, C, H, W)
        output: logits (L, B, Z_dim * Z_layers) or (L, Z_dim * Z_layers)

        where L is the length of the sequence, B is the batch size,
            C is the number of channels, H is the height, and W is the width.

        """
        # add the batch dimension if it doesn't exist
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        # (L, B, C, H, W) -> (L, B, Z_dim * Z_layers)
        logits = torch.stack(
            [self.encoder(x0).view(-1, self.z_layers * self.z_dim) for x0 in x]
        )

        # pass the sequence of logits through the LSTM
        # (L, B, Z_dim * Z_layers) -> (L, B, Z_dim * Z_layers)
        lstm_out, _ = self.lstm(logits)

        # skip connection
        logits = logits + lstm_out

        # re parameterize the logits
        # (L, B, Z_dim * Z_layers) -> (L, B, Z_layers, Z_dim)
        z = torch.stack(
            [
                self.reparameterize(l.reshape(-1, self.z_layers, self.z_dim))
                for l in logits
            ]
        )

        return logits, z

    def decode(self, z):
        """
        The decoder needs to handle batched or unbatched inputs, but should always assume
        a sequence.

        input: z (N, B, Z_dim, Z_layers) or (N, Z_dim, Z_layers)
        ouput: x (N, B, C, H, W) or (N, C, H, W)
        """
        # add the batch dimension if it doesn't exist
        if z.ndim == 2:
            z = z.unsqueeze(1)
        z = z.flatten(start_dim=2)

        # (N, B, Z_dim * Z_layers) -> (N, B, C, H, W)
        x_hat = torch.stack([self.decoder(z0) for z0 in z])

        raise x_hat

    def kl_loss(self, logits):
        """
        Logits are shape (L, B, N, K), where B is the number of batches, N is the number
            of categorical distributions, L is the length of the sequence, and K is the number of classes
        returns: kl-divergence, in nats
        """
        assert logits.ndim == 4
        L, B, N, K = logits.shape
        logits = logits.view(L * B * N, K)

        q = Categorical(logits=logits)
        p = Categorical(probs=torch.full((B * N, K), 1.0 / K).to(DEVICE))

        # sum loss over dimensions in each example, average over batch
        kl = kl_divergence(q, p).view(B, N).sum(1).mean()

        return kl

    def recontruction_loss(self, x, x_hat):
        mse_loss = F.mse_loss(x_hat, x, reduction="none")
        # sum loss over dimensions in each example, average over batch and sequence
        mse_loss = mse_loss.view(x.shape[0], x.shape[1], -1).sum(2).mean(1)
        return mse_loss

    def get_state(self, obs: Tensor):
        r"""
        Takes in observations and returns discrete states

        Args:
            obs (Tensor): a (N, C, H, W) or (N, B, C, H, W) tensor
            hidden_state (Tensor, optional) a NxD tensor of hidden states.  If
                no value is specified, will use a default value of zero
        """
        assert obs.ndim <= 5
        assert_correct_end_shape(obs, self.input_shape)
        raise NotImplementedError
        # hidden_state = (
        #     hidden_state
        #     if isinstance(hidden_state, Tensor)
        #     else torch.zeros_like(obs).to(DEVICE)
        # )

        # self.eval()
        # with torch.no_grad():
        #     # check the dimensions, expand if unbatch

        #     # expand if unbatched
        #     assert obs.view(-1).shape[0] % self.encoder.nin == 0
        #     print(obs.shape)
        #     if obs.view(-1).shape[0] == self.encoder.nin:
        #         obs = obs[None, ...]
        #         hidden_state = hidden_state[None, ...]

        #     state_vars = []
        #     for o, h in zip(obs, hidden_state):
        #         print(o, h)
        # #         _, z = self._encode_from_state(o.to(DEVICE), h.to(DEVICE))
        # #         state_vars.append(torch.argmax(z, dim=-1).detach().cpu().numpy())

        # # return state_vars
