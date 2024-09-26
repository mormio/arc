import importlib

import torch
from torch.utils.data import random_split


def get_solver(problem, solvers_module=None):
    if not solvers_module:
        solvers_module = importlib.import_module("arc.arcdsl.solvers")
    # get the function
    solver_function = getattr(solvers_module, f"solve_{problem}")
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


def load_data(dataset="ARC"):
    # TODO make this load either ARC dataset, REARC dataset, or synthetic dataset..
    pass
