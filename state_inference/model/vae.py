from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical

from state_inference.utils.pytorch_utils import (
    DEVICE,
    assert_correct_end_shape,
    check_shape_match,
    gumbel_softmax,
    maybe_expand_batch,
)

OPTIM_KWARGS = dict(lr=3e-4)
VAE_BETA = 1.0
VAE_TAU = 1.0
VAE_GAMMA = 1.0
INPUT_SHAPE = (1, 40, 40)


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)

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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(MLP):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.view(x.shape[0], -1))


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        act_fn="relu",
        conv_kwargs=None,
    ) -> None:
        super().__init__()
        conv_kwargs = conv_kwargs if conv_kwargs is not None else dict()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = getattr(F, act_fn)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class CnnEncoder(ModelBase):
    def __init__(
        self,
        in_channel: int,
        output_size: int,
        height: int = 40,
        width: int = 40,
        act_fn="relu",
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            Conv2dBlock(in_channel, 32, 8, stride=4, padding=0, act_fn=act_fn),
            Conv2dBlock(32, 64, 4, stride=2, padding=0, act_fn=act_fn),
            Conv2dBlock(64, 64, 3, stride=1, padding=0, act_fn=act_fn),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        self.eval()
        x = torch.rand(1, in_channel, height, width)
        with torch.no_grad():
            n_flatten = self.cnn(x).shape[-1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, output_size), nn.ReLU())

        self.input_shape = (in_channel, height, width)

    def forward(self, x):
        # Assume NxCxHxW input or CxHxW input
        assert_correct_end_shape(x, self.input_shape)
        x = maybe_expand_batch(x, self.input_shape)
        return self.linear(self.cnn(x))

    def encode_sequence(self, x: Tensor, batch_first: bool = True) -> Tensor:
        assert x.ndim == 5
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        x = torch.stack([self(xt) for xt in x])
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        return x


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

    def loss(self, x, target):
        y_hat = self(x)
        return F.mse_loss(y_hat, torch.flatten(target, start_dim=1))


class ConvTrans2dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        act_fn="relu",
        conv_kwargs=None,
    ) -> None:
        super().__init__()
        conv_kwargs = conv_kwargs if conv_kwargs is not None else dict()
        self.conv_trans2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = getattr(F, act_fn)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_trans2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class CnnDecoder(ModelBase):
    def __init__(self, input_size: int, channel_out: int, act_fn: str = "relu"):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(input_size, 4 * 4 * 128), nn.ReLU())

        self.deconv = nn.Sequential(
            ConvTrans2dBlock(128, 128, 4, stride=2, padding=1, act_fn=act_fn),
            ConvTrans2dBlock(128, 64, 4, stride=2, padding=1, act_fn=act_fn),
            ConvTrans2dBlock(64, 32, 4, stride=4, padding=0, act_fn=act_fn),
            ConvTrans2dBlock(32, channel_out, 3, stride=1, padding=1, act_fn=act_fn),
        )

    def forward(self, x):
        hidden = self.fc(x)
        hidden = hidden.view(-1, 128, 4, 4)  # does not effect batching
        hidden = self.deconv(hidden)
        observation = F.sigmoid(hidden)
        if observation.shape[0] == 1:
            return observation.squeeze(0)
        return observation

    def loss(self, x, target):
        y_hat = self(x)
        return F.mse_loss(y_hat, torch.flatten(target, start_dim=1))


def unit_test_vae_reconstruction(model, input_shape):
    # Unit Test
    model.eval()
    with torch.no_grad():
        x = torch.rand(*input_shape)
        _, z = model.encode(x)
        x_hat = model.decode(z)
        assert check_shape_match(
            x_hat, input_shape
        ), f"Reconstruction Shape {tuple(x_hat.shape)} Doesn't match Input {input_shape}"


class StateVae(ModelBase):
    def __init__(
        self,
        encoder: ModelBase,
        decoder: ModelBase,
        z_dim: int,
        z_layers: int,
        beta: float = VAE_BETA,
        tau: float = VAE_TAU,
        gamma: float = VAE_GAMMA,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
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
        self.input_shape = input_shape

        unit_test_vae_reconstruction(self, input_shape)

    def reparameterize(self, logits):
        """
        Assume input shape of NxLxD
        """
        assert logits.ndim == 3
        assert logits.shape[1] == self.z_layers
        assert logits.shape[2] == self.z_dim

        # either sample the state or take the argmax
        if self.training:
            z = gumbel_softmax(logits=logits, tau=self.tau, hard=False)
        else:
            s = torch.argmax(logits, dim=-1)  # tensor of n_batch * self.z_n_layers
            z = F.one_hot(s, num_classes=self.z_dim)
        return z

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Assume input shape of NxCxHxW
        """
        # reshape encoder output to (n_batch, z_layers, z_dim)
        logits = self.encoder(x).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def decode(self, z):
        return self.decoder(z.view(-1, self.z_layers * self.z_dim).float())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits, z = self.encode(x)
        x_hat = self.decode(z)
        return (logits, z), x_hat.view(x.shape)  # preserve original shape

    def kl_loss(self, logits):
        return Categorical(logits=logits).entropy().mean()

    def loss(self, x: Tensor) -> Tensor:
        x = x.to(DEVICE).float()
        (logits, _), x_hat = self(x)

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = F.mse_loss(x_hat, x)

        return recon_loss + kl_loss * self.beta

    def get_state(self, x):
        """
        Assume input shape of NxCxHxW or CxHxW.
        """
        assert x.ndim <= 4
        assert_correct_end_shape(x, self.input_shape)

        self.eval()
        with torch.no_grad():
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


class StateVaeLearnedTau(StateVae):
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
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
        self.tau = torch.nn.Parameter(torch.tensor([tau]), requires_grad=True)

    def anneal_tau(self):
        pass


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
    def __init__(
        self,
        encoder: ModelBase,
        decoder: ModelBase,
        next_obs_decoder: DecoderWithActions,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
        self.next_obs_decoder = next_obs_decoder

    def forward(self, x: Tensor):
        raise NotImplementedError

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        obs, actions, obsp = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()
        obsp = obsp.to(DEVICE).float()

        logits, z = self.encode(obs)
        z = z.view(-1, self.z_layers * self.z_dim).float()

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(z, obs)
        next_obs_loss = self.next_obs_decoder.loss(z, actions, obsp)

        return recon_loss + next_obs_loss + kl_loss * self.beta


class GruEncoder(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        self.feature_extracter = MLP(input_size, hidden_sizes, embedding_dims, dropout)
        self.gru_cell = nn.GRUCell(embedding_dims, embedding_dims, **gru_kwargs)
        self.hidden_encoder = nn.Linear(embedding_dims + n_actions, embedding_dims)
        self.action_encoder = nn.Linear(n_actions, n_actions)
        self.batch_first = batch_first
        self.hidden_size = embedding_dims
        self.n_actions = n_actions
        self.nin = input_size

    def rnn(self, obs: Tensor, h: Tensor) -> Tensor:
        print(obs.shape, h.shape)
        x = self.feature_extracter(obs)
        return self.gru_cell(x, h)

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
    ):
        obs = torch.flatten(obs, start_dim=2)
        if self.batch_first:
            obs = torch.permute(obs, (1, 0, 2))
            actions = torch.permute(actions, (1, 0, 2))

        n_batch = obs.shape[1]

        # initialize the hidden state and action
        h = torch.zeros(n_batch, self.hidden_size)
        a_prev = torch.zeros(n_batch, self.n_actions)

        # loop through the sequence of observations
        for o, a in zip(obs, actions):
            # encode the hidden state with the previous actions
            h = self.hidden_encoder(torch.concat([h, a_prev], dim=1).to(DEVICE))

            # encode the action for the next step
            a_prev = self.action_encoder(a)

            # pass the observation through the rnn (+ encoder)
            h = self.rnn(o, h)

        return h


class RecurrentVae(StateVae):
    def __init__(
        self,
        encoder: GruEncoder,
        decoder: ModelBase,
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
        recon_loss = self.decoder.loss(z, obs[:, -1, ...])

        return recon_loss + kl_loss * self.beta
