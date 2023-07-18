import numpy as np
import torch


def eval_model(model, n, start_state=None):
    env = model.get_env()
    env.env_method("set_initial_state", start_state)
    obs = env.reset()
    rewards = []
    obs_all = []
    state_trajectory = []
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=start_state is not None)
        obs, rew, done, info = env.step(action)
        state_trajectory.append(
            (info[0]["start_state"], action[0], info[0]["successor_state"], rew[0])
        )
        rewards.append(rew)
        obs_all.append(obs)
        if done:
            obs = env.reset()
    return np.array(rewards), obs_all, state_trajectory


def score_policy(
    model,
    optimal_policy,
    n_obs=10,
    n_states=400,
    map_height=60,
):
    env = model.get_env()
    score = []
    for _ in range(n_obs):
        obs = [
            torch.tensor(env.env_method("generate_observation", s)[0]).view(
                1, map_height, map_height
            )
            for s in range(n_states)
        ]
        obs = torch.stack(obs)
        with torch.no_grad():
            pmf = (
                model.policy.get_distribution(torch.tensor(obs))
                .distribution.probs.detach()
                .numpy()
            )
        score.append(np.sum(optimal_policy * pmf, axis=1))
    return np.array(score).mean(axis=0)


def sample_policy(model, n_states=400, map_height=60, cnn=True):
    env = model.get_env()
    # print(type(env.reset()))
    shape = [map_height, map_height]
    if cnn:
        shape = [1, map_height, map_height]

    obs = [
        np.array(env.env_method("generate_observation", s)[0]).reshape(*shape)
        for s in range(n_states)
    ]
    # print(obs.shape)
    return model.predict(np.stack(obs), deterministic=True)


def train_model(
    model,
    optimal_policy,
    n_epochs,
    n_train_steps,
    n_obs=5,
    n_states=400,
    map_height=60,
    n_eval_steps=100,
    test_start_state=None,
):
    model_reward, score = [], []
    for e in range(n_epochs):
        rew, _, state_trajectory = eval_model(model, n_eval_steps, test_start_state)
        score.append(score_policy(model, optimal_policy, n_obs, n_states, map_height))
        model_reward.append(rew.sum())

        s = e * n_train_steps
        print(
            f"Training_steps {s}, reward {model_reward[-1]}, score {np.mean(score[-1])}"
        )
        model.learn(total_timesteps=n_train_steps, progress_bar=False)

    return model_reward, score
