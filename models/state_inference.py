from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# helper functions for training
QUIET = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        # self.net.pop()
        self.net.pop()
        self.net.pop()

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


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
            z = F.gumbel_softmax(logits=logits, tau=self.tau, hard=False)
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
