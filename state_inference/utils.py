from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from sklearn.neural_network import MLPClassifier

from state_inference.env import ObservationModel
from state_inference.pytorch_utils import DEVICE


class StateReconstruction:
    def __init__(
        self,
        vae_model: nn.Module,
        observation_model: ObservationModel,
        train_states: List[int],
    ):
        n = len(train_states)
        train_observations = torch.stack(observation_model(train_states)).view(n, -1)

        # encodes the states
        z_train = vae_model.get_state(train_observations.to(DEVICE))

        X_train = z_train.view(n, -1).detach().cpu().numpy()

        # self.clf = LogisticRegression(solver="lbfgs").fit(X_train, train_states)
        self.clf = MLPClassifier(
            hidden_layer_sizes=[100, 100, 100], learning_rate_init=3e-4
        ).fit(X_train, train_states)

        self.observation_model = observation_model
        self.vae_model = vae_model

    def _embed(self, states: List[int]):
        n = len(states)
        obs_test = torch.stack(self.observation_model(states)).view(n, -1)
        embed_state_vars = self.vae_model.get_state(obs_test.to(DEVICE))
        embed_state_vars = embed_state_vars.view(n, -1).detach().cpu().numpy()
        return embed_state_vars

    # def _internal_log_predict(self, states: List[])

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
        return np.exp(self.log_loss_by_time(pred_log_probs, states))

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
