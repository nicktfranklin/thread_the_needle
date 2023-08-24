import yaml


def parse_model_config(model_name, config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config[model_name]


def parse_task_config(task, config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

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
