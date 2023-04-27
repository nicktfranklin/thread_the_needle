from typing import List

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from state_inference.env import ObservationModel
from state_inference.model import mDVAE


class StateReconstruction:
    def __init__(
        self,
        vae_model: mDVAE,
        observation_model: ObservationModel,
        train_states: List[int],
    ):
        n = len(train_states)
        train_observations = torch.stack(observation_model(train_states)).view(n, -1)

        # encodes the states
        logits_train, _ = vae_model.encode_states(train_observations)

        X_train = logits_train.view(n, -1).detach().cpu().numpy()

        self.clf = LogisticRegression(max_iter=10000).fit(X_train, train_states)

        self.observation_model = observation_model
        self.vae_model = vae_model

    def _embed(self, states: List[int]):
        n = len(states)
        obs_test = torch.stack(self.observation_model(states)).view(n, -1)
        embed_logits_test, _ = self.vae_model.encode_states(obs_test)
        embed_logits_test = embed_logits_test.view(n, -1).detach().cpu().numpy()
        return embed_logits_test

    def predict_log_prob(self, states: List[int]):
        # returns (n_samples, n_classes) matrix of log probs
        return normalize_log_probabilities(
            self.clf.predict_log_proba(self._embed(states))
        )

    def predict_prob(self, states: List[int]):
        return self.clf.predict_proba(self._embed(states))

    @staticmethod
    def log_loss_by_time(pred_log_probs: np.ndarray, states: List[int]):
        return np.array([ps[s] for ps, s in zip(*[pred_log_probs, states])])

    def cross_entropy(self, pred_log_probs: np.ndarray, states: List[int]):
        return -self.log_loss_by_time(pred_log_probs, states).mean()

    def accuracy(self, pred_log_probs: np.ndarray, states: List[int]):
        return np.exp(self.log_loss_by_time(pred_log_probs, states)).mean()

    @staticmethod
    def entropy(log_probability: np.ndarray):
        return -np.sum([log_probability * np.exp(log_probability)], axis=1)


def normalize_log_probabilities(log_probs: np.ndarray) -> np.ndarray:
    return log_probs - logsumexp(log_probs, axis=1).reshape(-1, 1)


def logdotexp(A, B):
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C


def forward_pass(
    log_observation: np.ndarray, log_transitions: np.ndarray
) -> np.ndarray:
    """
    log_observation: (n_obs, n_states)
    log_transitions: (n_states, n_states)
    """

    filter_posterior = np.copy(log_observation)

    for ii, p in enumerate(filter_posterior):
        idx = ii + 1
        if ii < log_observation.shape[0] - 1:
            prior = logdotexp(log_transitions, p)
            filter_posterior[idx] += prior
            filter_posterior[idx] -= logsumexp(filter_posterior[idx])  # normalize

    return filter_posterior


def backwards_pass(
    log_observation: np.ndarray, log_transitions: np.ndarray
) -> np.ndarray:
    """
    log_observation: (n_obs, n_states)
    log_transitions: (n_states, n_states)
    """
    n_obs = log_observation.shape[0]
    filter_posterior = np.copy(log_observation)

    # for ii, p in enumerate(filter_posterior):
    for ii in range(n_obs - 1, 0, -1):
        prior = logdotexp(log_transitions, filter_posterior[ii])
        filter_posterior[ii - 1] += prior
        filter_posterior[ii - 1] -= logsumexp(filter_posterior[ii - 1])  # normalize

    return filter_posterior


def BayesianSmoothing(log_observation: np.ndarray, log_transitions: np.ndarray):
    """
    this model assumes symetric transitions
    """
    return normalize_log_probabilities(
        forward_pass(log_observation, log_transitions)
        + backwards_pass(log_observation, log_transitions)
    )


def BayesianFilter(log_observation: np.ndarray, log_transitions: np.ndarray):
    return forward_pass(log_observation, log_transitions)


# Note: there is an issue F.gumbel_softmax that appears to causes an error w
# where a valide distribution will return nans, preventing training.  Fix from
# https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
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
