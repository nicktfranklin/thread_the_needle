from typing import Dict, Hashable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.categorical import Categorical

from state_inference.utils import gumbel_softmax

# helper functions for training
QUIET = False

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


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

    def encode_states(
        self, observations: Union[np.array, torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(observations).to(DEVICE).float()
            logits, z = self.encode(x)
        return logits, z


def train(model, train_loader, optimizer, clip_grad=None):
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
        train_losses.extend(train(model, train_loader, optimizer, grad_clip))
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)

        model.anneal_tau()

        if not QUIET:
            print(f"Epoch {epoch}, ELBO Loss (test) {test_loss:.4f}")

    return train_losses, test_losses


class TransitionEstimator:
    ## Note: does not take in actions

    def __init__(self):
        self.transitions = dict()
        self.pmf = dict()

    def update(self, s: Hashable, sp: Hashable):
        if s in self.transitions:
            if sp in self.transitions[s]:
                self.transitions[s][sp] += 1
            else:
                self.transitions[s][sp] = 1
        else:
            self.transitions[s] = {sp: 1}

        N = float(sum(self.transitions[s].values()))
        self.pmf[s] = {sp0: v / N for sp0, v in self.transitions[s].items()}

    def batch_update(self, list_states: List[Hashable]):
        for ii in range(len(list_states) - 1):
            self.update(list_states[ii], list_states[ii + 1])

    def get_transition_probs(self, s: Hashable) -> Dict[Hashable, float]:
        # if a state is not in the model, assume it's self-absorbing
        if s not in self.pmf:
            return {s: 1.0}
        return self.pmf[s]


class RewardEstimator:
    def __init__(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():  # pylint: disable=consider-iterating-dictionary
            self.counts[s] += np.array([r, 1])
        else:
            self.counts[s] = np.array([r, 1])

        self.state_reward_function[s] = self.counts[s][0] / self.counts[s][1]

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_reward(self, state):
        return self.state_reward_function[state]


def value_iteration(
    t: Dict[Union[str, int], TransitionEstimator],
    r: RewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _inner_sum(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * v[sp]
        return _sum

    def _expected_reward(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * r.get_reward(sp)
        return _sum

    for _ in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = _expected_reward(s, a) + gamma * _inner_sum(s, a)
        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v


class RewardModel:
    # Generative Model
    pass
