from argparse import ArgumentParser
import numpy as np
from itertools import product
from ast import literal_eval


def parse_args(args: list):
    parsey = ArgumentParser()
    parsey.add_argument("-c", "--command-string", type=str, required=True)
    return parsey.parse_args(args)


def iter_wrapper(name, iterator):
    for v in iterator:
        yield {name: v}


def str_to_py(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError):
        return str(s)


def uniform(low, high, num):
    return np.random.uniform(low, high, (num,)).tolist()


def loguniform(low, high, num):
    log_low = np.log(low)
    log_high = np.log(high)
    log_samples = np.random.uniform(log_low, log_high, (num,))
    samples = np.exp(log_samples).tolist()
    return samples


def main(args):
    """
    Example: -c kp_0.4:
    """
    args = parse_args(args)
    fn_mapping = {
        "range": range,
        "linspace": np.linspace,
        "in": lambda *x: list(x),
        "uniform": uniform,
        "loguniform": loguniform,
    }
    iterators = []

    sub_commands = args.command_string.split("--")
    for command in sub_commands:
        parts = command.split("/")
        variable_name, fn_name, *args = parts
        fn = fn_mapping[fn_name]
        args = [str_to_py(arg) for arg in args]
        iterators.append(iter_wrapper(variable_name, fn(*args)))

    # Now for the main loop
    for variables in product(*iterators):
        kwargs = {}
        for var in variables:
            kwargs.update(var)
        yield kwargs