import yaml


def load_config(file_name):
    with open(file_name) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def parse_model_config(model_name, config_file):
    config = load_config(config_file)
    return config[model_name]


def parse_task_config(task_name, config_file):
    config = load_config(config_file)
    return config[task_name]["env_kwargs"]
