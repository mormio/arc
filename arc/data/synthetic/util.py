import os
import json
import importlib
import torch
from torch.utils.data import random_split
from arc.functions_library import FUNCTION_CLASSES


def load_synthetic_data():

    home_dir = os.environ["HOME"]
    data_dir = os.path.join(home_dir, "arc", "arc", "data", "synthetic")
    with open(os.path.join(data_dir, "synthetic_data.json"), "r") as f:
        synthetic_data = json.load(f)

    return synthetic_data


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
