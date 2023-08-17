import pandas as pd
from stable_baselines3 import PPO

from state_inference.gridworld_env import CnnWrapper, ThreadTheNeedleEnv
from state_inference.utils.training_utils import parse_task_config, train_model

CONFIG_FILE = "state_inference/env_config.yml"
TASK_NAME = "thread_the_needle"
TASK_CLASS = ThreadTheNeedleEnv
OUTPUT_FILE_NAME = "ThreadTheNeedleSims.csv"


def train():
    env_kwargs, training_kwargs = parse_task_config(TASK_NAME, CONFIG_FILE)

    # create the task
    task = CnnWrapper(TASK_CLASS.create_env(**env_kwargs))

    pi, _ = task.get_optimal_policy()
    training_kwargs["optimal_policy"] = pi

    results = []

    def append_res(results, rewards, model_name):
        results.append(
            {
                "Rewards": rewards,
                "Model": [model_name] * (training_kwargs["n_epochs"] + 1),
                "Epoch": [ii for ii in range(training_kwargs["n_epochs"] + 1)],
            }
        )

    ppo = PPO("CnnPolicy", task, verbose=0)
    ppo_rewards = train_model(ppo, **training_kwargs)
    append_res(results, ppo_rewards, "PPO")

    results = pd.concat([pd.DataFrame(res) for res in results])
    results.set_index("Epoch").to_csv(OUTPUT_FILE_NAME)


if __name__ == "__main__":
    train()
