from gridworld import OpenEnv, ThreadTheNeedleEnv


def make(task_name, *args, **kwargs):
    if task_name == "thread_the_needle":
        return ThreadTheNeedleEnv(*args, **kwargs)
    elif task_name == "open":
        return OpenEnv(*args, **kwargs)
    else:
        raise ValueError("Unknown task name: {}".format(task_name))
