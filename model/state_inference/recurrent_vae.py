from typing import List

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

    def encode(self, x):
        raise NotImplementedError

    def _encode_from_sequence(self, obs: Tensor, actions: Tensor) -> Tensor:
        logits = self.encoder(obs, actions)
        z = self.reparameterize(logits)
        return logits, z

    def _encode_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        logits = self.encoder.rnn(obs, h)
        z = self.reparameterize(logits)
        return logits, z

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
        (obs, actions), _ = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()

        logits, z = self._encode_from_sequence(obs, actions)  # this won't work
        z = z.view(-1, self.z_layers * self.z_dim).float()

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder_loss(z, obs[:, -1, ...])

        return recon_loss + kl_loss * self.beta
