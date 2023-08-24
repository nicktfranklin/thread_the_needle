import yaml


def load_config(file_name):
    with open(file_name) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def parse_model_config(model_name, config_file):
    config = load_config(config_file)
    return config[model_name]


def parse_task_config(task, config_file):
    config = load_config(config_file)

    env_kwargs = config[task]["env_kwargs"]

    training_kwargs = config["training_kwargs"]
    training_kwargs.update(
        dict(
            n_states=env_kwargs["n_states"],
            map_height=env_kwargs["map_height"],
            test_start_state=config[task]["test_start_state"],
        )
    )

    return env_kwargs, training_kwargs
