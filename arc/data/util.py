import importlib

import torch
from torch.utils.data import random_split

from .re_arc.util import load_data as load_rearc_dataset
from .real.util import load_data as load_arc_dataset


def get_solver(problem, solvers_module=None):
    if not solvers_module:
        solvers_module = importlib.import_module("arc.arcdsl.solvers")
    # get the function
    try:
        solver_function = getattr(solvers_module, f"solve_{problem}")
    except Exception as e:
        print(f"Solver not found for problem {problem}. {e}")
        return None

    return solver_function


def split_dataset(dataset, val_split=0.2, seed=0):
    """
    Splits a dataset into training and validation sets.
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Calculate the split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def load_data(dataset="ARC", **kwargs):
    if dataset == "ARC":
        data = load_arc_dataset(**kwargs)
    elif dataset == "REARC":
        data = load_rearc_dataset(**kwargs)
    # TODO: elif dataset == "synthetic"

    return data
