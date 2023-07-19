from typing import Union

import numpy as np
import scipy.sparse
from scipy.special import logsumexp


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


def one_hot(a, num_classes):
    ### define simple deterministic transition functions using cardinal movements
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def inverse_cmf_sampler(pmf: Union[np.ndarray, scipy.sparse.csr_matrix]) -> int:
    if type(pmf) == scipy.sparse.csr_matrix:
        pmf = pmf.toarray()

    return np.array(np.cumsum(np.array(pmf)) < np.random.rand(), dtype=int).sum()
