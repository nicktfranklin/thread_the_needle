from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# helper functions for training
QUIET = False

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


import torch
import torch.nn.functional as F


# Note: there is an issue F.gumbel_softmax that appears to causes an error w
# where a valide distribution will return nans, preventing training.  Fix from
# https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.nin = input_size
        self.nout = output_size

        # define a simple MLP neural net
        self.net = []
        hs = [self.nin] + hidden_sizes + [self.nout]
        for h0, h1 in zip(hs, hs[1:]):
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

    def forward(self, x):
        return self.net(x)


class Encoder(MLP):
    pass


class Decoder(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__(input_size, hidden_sizes, output_size, dropout)
        self.net.pop(-1)


class mDVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
        random_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_layers = z_layers
        self.z_dim = z_dim
        self.beta = beta
        self.tau = tau
        self.gamma = gamma
        self.random_encoder = random_encoder

        if random_encoder:
            for child in self.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False

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

    def forward(self, x):
        _, z = self.encode(x)
        return self.decode(z)

    def kl_loss(self, logits):
        return Categorical(logits=logits).entropy().mean()

    def reconstruction_loss(self, x, z):
        x_hat = self.decode(z)
        return F.mse_loss(x_hat, x)

    def loss(self, x):
        logits, z = self.encode(x)
        return self.loss_from_embedding(x, logits, z)

    def loss_from_embedding(self, x, logits, z):
        kl_loss = self.kl_loss(logits)
        recon_loss = self.reconstruction_loss(x, z)
        return recon_loss + kl_loss * self.beta

    def state_probability(self, x):
        with torch.no_grad():
            logits = self.encoder(x)
            return Categorical(logits=logits).probs

    def get_state(self, x):
        self.eval()
        _, z = self.encode(x)
        return torch.argmax(z, dim=-1)

    def decode_state(self, s: Tuple[int]):
        self.eval()
        z = F.one_hot(torch.tensor(s).to(DEVICE), self.z_dim).view(-1).unsqueeze(0)
        with torch.no_grad():
            return self.decode(z).detach().cpu().numpy()

    def anneal_tau(self):
        self.tau *= self.gamma


def train(model, train_loader, optimizer, epoch, clip_grad=None):
    model.train()

    train_losses = []
    for x in train_loader:
        x = x.to(DEVICE).float()

        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(DEVICE).float()
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args["epochs"], train_args["lr"]
    grad_clip = train_args.get("grad_clip", None)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    test_losses = [eval_loss(model, test_loader)]
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch, grad_clip))
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)

        model.anneal_tau()

        if not QUIET:
            print(f"Epoch {epoch}, ELBO Loss (test) {test_loss:.4f}")

    return train_losses, test_losses

def encode_states(model: mDVAE, observations: np.array):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(observations).to(DEVICE).float()
        logits, z = model.encode(x)
    return logits, z
