from model.value_iteration import ValueIterationAgent


class DynaAgent(ValueIterationAgent):
    dyna_epochs = 10

    def estimate_offline(self):
        # resetimate the model from the new states
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        s, sp = self._precalculate_states_for_batch_training()
        for idx, obs in enumerate(self.rollout_buffer.get_all()):
            self.transition_estimator.update(s[idx], obs.a, sp[idx])
            self.reward_estimator.update(sp[idx], obs.r)

        # resample with dyna
        self._dyna_updates(self.dyna_epochs * len(s))

        # estimate value functions
        self.value_function = {s0: max(self.policy.q_values.values()) for s0 in s}
