import ast
import importlib
import inspect

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


def string_grid_to_ascii_art(grid: str):
    """Turn a string from the ARC dataset, representing a grid, into ascii."""

    def _string_to_list_of_lists(s):
        try:
            # Use ast.literal_eval to safely evaluate the string as a Python literal
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            raise ValueError(
                "Invalid input string. Must be a valid representation of a list of lists."
            )

    # turn into bona fide python list of lists
    try:
        grid = _string_to_list_of_lists(grid)
    except ValueError as e:
        print(e)
        return None

    # Define a mapping of numbers to ASCII characters
    ascii_chars = " .:-=+*#%@"

    # Convert each number to its corresponding ASCII character
    ascii_art = []
    for row in grid:
        ascii_row = "".join(ascii_chars[num] for num in row)
        ascii_art.append(ascii_row)

    # Join the rows and return the result
    return "\n".join(ascii_art)
