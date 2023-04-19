from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch import nn

from environments.state_inference import ObservationModel


class StateReconstruction:
    def __init__(
            self,
            vae_model: nn.Module, 
            observation_model: ObservationModel,
            train_states: List[int],
        ):
        n = len(train_states)
        train_observations = np.array(observation_model(train_states)).reshape(n, -1)

        # encodes the states
        logits_train, _ = vae_model.encode_states(train_observations)

        X_train = logits_train.view(n, -1).detach().cpu().numpy()

        self.clf = LogisticRegression().fit(X_train, train_states)
        self.observation_model = observation_model
        self.vae_model = vae_model

    def _embed(self, states: List[int]):
        n = len(states)
        obs_test = np.array(self.observation_model(states)).reshape(n, -1)
        embed_logits_test, _ = self.vae_model.encode_states(obs_test)
        embed_logits_test = embed_logits_test.view(n, -1).detach().cpu().numpy()
        return embed_logits_test

    def predict_log_prob(self, states: List[int]):
        return self.clf.predict_log_proba(self._embed(states))

    def predict_prob(self, states: List[int]):
        return self.clf.predict_proba(self._embed(states))

    def score(self, states: List[int]):
        return np.mean([
            ps[s] for ps, s in zip(*[self.predict_prob(states), states])
        ])

    @staticmethod
    def entropy(log_probability: np.ndarray):
        return -np.sum([
            log_probability * np.exp(log_probability)
        ], axis=1)
    