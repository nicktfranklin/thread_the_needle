import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import trange

from utils.pytorch_utils import (
    DEVICE,
    assert_correct_end_shape,
    check_shape_match,
    gumbel_softmax,
    maybe_expand_batch,
    train,
)

OPTIM_KWARGS = dict(lr=3e-4)
VAE_BETA = 1.0
VAE_TAU = 1.0
VAE_TAU_ANNEALING_RATE = 1.0
INPUT_SHAPE = (1, 40, 40)


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
    ):
        super().__init__()
        self.nin = input_size
        self.nout = output_size

        # define a simple MLP neural net
        self.net = []

        d_in = self.nin
        for d_out in hidden_sizes:
            self.net.extend(
                [
                    nn.Linear(d_in, d_out),
                    nn.BatchNorm1d(d_out),
                    nn.ELU(),
                ]
            )
            d_in = d_out

        # final linear layer
        self.net.append(nn.Linear(d_out, self.nout))

        self.net = nn.Sequential(*self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BaseEncoder(nn.Module):
    pass


class MlpEncoder(BaseEncoder):
    def __init__(
        self,
        input_shape: int | Tuple[int],
        hidden_sizes: List[int],
        embedding_dim: int,
    ):
        super().__init__()
        if isinstance(input_shape, tuple):
            input_shape = torch.tensor(input_shape).prod().item()

        self.net = MLP(input_shape, hidden_sizes, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.view(x.shape[0], -1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


class CnnEncoder(BaseEncoder):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        input_shape: Tuple[int, int, int],
        channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512]

        modules = []
        h_in = in_channels
        for h_dim in channels:
            modules.append(ConvBlock(h_in, h_dim, 3, stride=2, padding=1))
            h_in = h_dim
        self.cnn = nn.Sequential(*modules)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels[-1] * 4),
            nn.Linear(channels[-1] * 4, channels[-1] * 4),
            nn.ELU(),
            nn.Linear(channels[-1] * 4, embedding_dim),
        )

        assert in_channels == input_shape[0], "Input channels do not match shape!"

        self.input_shape = input_shape

    def forward(self, x):
        # Assume NxCxHxW input or CxHxW input
        assert_correct_end_shape(x, self.input_shape)
        x = maybe_expand_batch(x, self.input_shape)
        x = self.cnn(x)
        x = self.mlp(torch.flatten(x, start_dim=1))
        return x

    def encode_sequence(self, x: Tensor, batch_first: bool = True) -> Tensor:
        assert x.ndim == 5
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        x = torch.stack([self(xt) for xt in x])
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        return x


class BaseDecoder(nn.Module):
    def loss(self, x, target):
        y_hat = self(x)
        return F.mse_loss(y_hat, torch.flatten(target, start_dim=1))


class MlpDecoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        output_shape: int | Tuple[int, int, int],
    ):
        super().__init__()

        if isinstance(output_shape, tuple):
            d_out = torch.tensor(output_shape).prod().item()
        else:
            d_out = output_shape

        self.net = MLP(embedding_dim, hidden_sizes, d_out)
        self.output_shape = output_shape

    def forward(self, x):
        output = F.sigmoid(self.net(x))
        return output.view(output.shape[0], *self.output_shape)


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        output_padding=0,
    ):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t(x)
        x = self.act(x)
        return x


class CnnDecoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim: int,
        channel_out: int,
        channels: Optional[List[int]] = None,
        output_shape=None,  # not used
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512][::-1]

        self.fc = nn.Linear(embedding_dim, 4 * channels[0])
        self.first_channel_size = channels[0]

        modules = []
        for ii in range(len(channels) - 1):
            modules.append(
                ConvTransposeBlock(
                    channels[ii],
                    channels[ii + 1],
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        self.deconv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1],
                channels[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(channels[-1]),
            nn.GELU(),
            nn.Conv2d(channels[-1], out_channels=channel_out, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        hidden = self.fc(z)
        hidden = hidden.view(
            -1, self.first_channel_size, 2, 2
        )  # does not effect batching
        hidden = self.deconv(hidden)
        observation = self.final_layer(hidden)
        if observation.shape[0] == 1:
            return observation.squeeze(0)
        return observation


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


class StateVae(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        z_dim: int,
        z_layers: int,
        beta: float = VAE_BETA,
        tau: float = VAE_TAU,
        tau_annealing_rate: float = VAE_TAU_ANNEALING_RATE,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        tau_is_parameter: bool = False,
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
        self.tau_annealing_rate = tau_annealing_rate
        self.input_shape = input_shape

        if tau_is_parameter:
            self.tau = torch.nn.Parameter(torch.tensor([tau]), requires_grad=True)
            assert (
                self.tau_annealing_rate == 1.0
            ), "Must set tau annealing to 1 for learnable tau"

        unit_test_vae_reconstruction(self, input_shape)

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)

        return optimizer

    @classmethod
    def make_from_configs(cls, vae_config: Dict[str, Any], env_kwargs: Dict[str, Any]):
        h = env_kwargs["map_height"]
        input_shape = (1, h, h)

        vae_kwargs = vae_config["vae_kwargs"]
        vae_kwargs["input_shape"] = input_shape

        Encoder = getattr(sys.modules[__name__], vae_config["encoder_class"])
        Decoder = getattr(sys.modules[__name__], vae_config["decoder_class"])

        encoder_kwargs = vae_config["encoder_kwargs"]
        decoder_kwargs = vae_config["decoder_kwargs"]

        embedding_dim = vae_kwargs["z_dim"] * vae_kwargs["z_layers"]
        encoder_kwargs["embedding_dim"] = embedding_dim
        decoder_kwargs["embedding_dim"] = embedding_dim

        encoder_kwargs["input_shape"] = input_shape
        decoder_kwargs["output_shape"] = input_shape

        encoder = Encoder(**encoder_kwargs)
        decoder = Decoder(**decoder_kwargs)

        return cls(encoder, decoder, **vae_kwargs)

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
        """
        Logits are shape (B, N, K), where B is the number of batches, N is the number
            of categorical distributions and where K is the number of classes
        # returns kl-divergence, in nats
        """
        assert logits.ndim == 3
        B, N, K = logits.shape
        logits = logits.view(B * N, K)

        q = Categorical(logits=logits)
        p = Categorical(probs=torch.full((B * N, K), 1.0 / K).to(DEVICE))

        # sum loss over dimensions in each example, average over batch
        kl = dist.kl.kl_divergence(q, p).view(B, N).sum(1).mean()

        return kl

    def recontruction_loss(self, x, x_hat):
        mse_loss = F.mse_loss(x_hat, x, reduction="none")
        # sum loss over dimensions in each example, average over batch
        return mse_loss.view(x.shape[0], -1).sum(1).mean()

    def loss(self, x: Tensor) -> Tensor:
        x = x.to(DEVICE).float()
        (logits, _), x_hat = self(x)

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.recontruction_loss(x, x_hat)

        return recon_loss + kl_loss * self.beta

    def diagnose_loss(self, x):
        with torch.no_grad():
            x = x.to(DEVICE).float()
            (logits, _), x_hat = self(x)

            # get the two components of the ELBO loss
            kl_loss = self.kl_loss(logits)
            recon_loss = self.recontruction_loss(x, x_hat)

            loss = recon_loss - kl_loss * self.beta
            print(
                f"Total Loss: {loss:.4f}, Reconstruction: {recon_loss:.4f}, "
                + f"KL-Loss: {kl_loss:.4f}, beta: {self.beta}"
            )

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
        if self.tau_annealing_rate - 1 < 1e-4:
            return
        self.tau *= self.tau_annealing_rate

    def prep_next_batch(self):
        self.anneal_tau()

    def train_epochs(
        self,
        n_epochs,
        data_loader,
        optim,
        grad_clip: bool = False,
        progress_bar: bool = False,
    ):
        self.train()

        if progress_bar:
            iterator = trange(n_epochs, desc="Vae Epochs")
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            train(self, data_loader, optim, grad_clip)
            self.prep_next_batch()


class DecoderWithActions(BaseDecoder):
    def __init__(
        self,
        embedding_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        ouput_size: int,
    ):
        super().__init__()
        self.mlp = MlpDecoder(
            embedding_size,
            hidden_sizes,
            ouput_size,
        )
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
        encoder: BaseEncoder,
        decoder: BaseDecoder,
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


class GruEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        self.feature_extracter = MLP(input_size, hidden_sizes, embedding_dims)
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
        encoder: BaseEncoder,
        decoder: BaseDecoder,
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
