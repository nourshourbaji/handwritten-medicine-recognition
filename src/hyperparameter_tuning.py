import itertools
import math
import random

import pandas as pd

from src.train import run_experiment


def summarize(params, result):
    return {
        **params,
        "best_val_acc": result["best_val_acc"],
        "best_epoch": result["best_epoch"],
        "train_time_min": result["train_time_min"],
        "name": result["name"],
    }


def grid_search(base_config, param_grid, prefix, **run_args):
    """Try every combination of the values in param_grid.

    param_grid example: {"lr": [1e-2, 1e-3], "batch_size": [16, 32]} -> 4 runs
    """
    keys = list(param_grid)
    rows = []

    for values in itertools.product(*param_grid.values()):
        # zip(["lr","batch_size"], (0.001, 32)) → {"lr": 0.001, "batch_size": 32}
        params = dict(zip(keys, values))
        # build a name for example → "A_grid_lr=0.001_batch_size=32"
        name = prefix + "_" + "_".join(f"{k}={v}" for k, v in params.items())

        result = run_experiment(name, {**base_config, **params}, **run_args)
        rows.append(summarize(params, result))

    return pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)


def sample_params(param_space, rng):
    """A list -> pick one value. A tuple (low, high) -> sample on a log scale."""
    params = {}
    for key, space in param_space.items():
        if isinstance(space, tuple):
            low, high = math.log10(space[0]), math.log10(space[1])
            params[key] = float(f"{10 ** rng.uniform(low, high):.2e}")
        else:
            params[key] = rng.choice(space)
    return params


def random_search(base_config, param_space, n_trials, prefix, seed=42, **run_args):
    """Try n_trials random combinations sampled from param_space.

    param_space example: {"lr": (1e-4, 1e-2), "dropout": [0.0, 0.3, 0.5]}
    """
    rng = random.Random(seed)
    rows = []

    for trial in range(n_trials):
        params = sample_params(param_space, rng)
        name = f"{prefix}_trial{trial:02d}"

        print(f"\nTrial {trial + 1}/{n_trials}: {params}")
        result = run_experiment(name, {**base_config, **params}, **run_args)
        rows.append(summarize(params, result))

    return pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
