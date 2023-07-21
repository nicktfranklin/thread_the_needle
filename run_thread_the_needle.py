import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import A2C, DQN, PPO

from state_inference.gridworld_env import CnnWrapper, ThreadTheNeedleEnv
from state_inference.utils.training_utils import train_model

CONFIG_FILE = 'state_inference/env_config.yml'
TASK_NAME = 'thread_the_needle'
TASK_CLASS = ThreadTheNeedleEnv
OUTPUT_FILE_NAME = 'ThreadTheNeedleSims.csv'


def train():
    # parse the config file
    with open(CONFIG_FILE) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    env_kwargs = config[TASK_NAME]['env_kwargs']
    obs_kwargs = config['obs_kwargs']
    test_start_state = config[TASK_NAME]['test_start_state']
    training_kwargs = config['training_kwargs']

    # create the task
    task = CnnWrapper(TASK_CLASS.create_env(**env_kwargs, observation_kwargs=obs_kwargs))

    pi, _ = task.get_optimal_policy()

    train_kwargs = dict(
        optimal_policy=pi,
        n_epochs=training_kwargs['n_epochs'],
        n_train_steps=training_kwargs['n_steps'],
        n_obs=5,
        n_states=env_kwargs['n_states'],
        map_height=env_kwargs['map_height'],
        n_eval_steps=training_kwargs['n_eval_steps'],
        test_start_state=test_start_state,
    )



    results = []
    def append_res(results, rewards, model_name):
        results.append(
            {"Rewards": rewards,
             'Model': [model_name] * (config['training_kwargs']['n_epochs'] + 1),
             "Epoch": [ii for ii in range(config['training_kwargs']['n_epochs'] + 1)],
             }
        )


    ppo = PPO("CnnPolicy", task, verbose=0)
    ppo_rewards = train_model(ppo, **train_kwargs)
    append_res(results, ppo_rewards, 'PPO')

    # a2c = A2C("CnnPolicy", task, verbose=0)
    # a2c_rewards = train_model(a2c, **train_kwargs)

    # dqn = DQN("CnnPolicy", task, verbose=0)
    # dqn_rewards = train_model(dqn, **train_kwargs)

    results = pd.concat([pd.DataFrame(res) for res in results])
    results.set_index("Epoch").to_csv(OUTPUT_FILE_NAME)


if __name__ == "__main__":
    train()
