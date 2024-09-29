import importlib
import inspect

import numpy as np
import torch
from torch.utils.data import random_split

from arc.arcdsl import PRIMITIVES
from arc.arcdsl import solvers as solvers_mod

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


def get_primitives_vector_for_problem(problem_id, mod=None):
    if not mod:
        mod = solvers_mod

    solver = get_solver(problem_id, mod)
    # if theres no solver then get_solver() returns None
    if solver:
        solver_source = inspect.getsource(solver)

        # check which primitives are in the solver
        # make it into the label vector
        label = [0] * len(PRIMITIVES)
        for i, prim in enumerate(PRIMITIVES):
            if prim in solver_source:
                label[i] = 1
    else:
        label = None

    return label


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


def grid_to_ascii(grid: np.ndarray, separator: str = "|"):
    """Turn a string from the ARC dataset, representing a grid, into ascii."""

    return "\n".join(separator.join(str(x) for x in row) for row in grid)
