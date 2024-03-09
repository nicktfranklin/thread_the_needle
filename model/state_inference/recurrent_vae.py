from typing import List, Optional

import torch
from torch import Tensor, nn

from model.state_inference.vae import StateVae
from utils.pytorch_utils import DEVICE


class RecurrentVae(StateVae):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
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

        input: z (N, B, Z_dim * Z_layers) or (N, Z_dim * Z_layers)
        ouput: x (N, B, C, H, W) or (N, C, H, W)
        """
        raise NotImplementedError()

    def _encode_from_sequence(self, obs: Tensor, actions: Tensor) -> Tensor:
        raise NotImplementedError()
        # logits = self.encoder(obs, actions)
        # z = self.reparameterize(logits)
        # return logits, z

    def _encode_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        raise NotImplementedError()
        # logits = self.encoder.rnn(obs, h)
        # z = self.reparameterize(logits)
        # return logits, z

    def get_state(self, obs: Tensor, hidden_state: Optional[Tensor] = None):
        r"""
        Takes in observations and returns discrete states

        Args:
            obs (Tensor): a NxCxHxW tensor
            hidden_state (Tensor, optional) a NxD tensor of hidden states.  If
                no value is specified, will use a default value of zero
        """
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

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        raise NotImplementedError()
        # (obs, actions), _ = batch_data
        # obs = obs.to(DEVICE).float()
        # actions = actions.to(DEVICE).float()

        # logits, z = self._encode_from_sequence(obs, actions)  # this won't work
        # z = z.view(-1, self.z_layers * self.z_dim).float()

        # # get the two components of the ELBO loss
        # kl_loss = self.kl_loss(logits)
        # recon_loss = self.decoder_loss(z, obs[:, -1, ...])

        # return recon_loss + kl_loss * self.beta
