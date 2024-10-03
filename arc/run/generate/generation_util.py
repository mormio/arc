import inspect
from typing import Callable, Dict, get_type_hints

import numpy as np
from type_util import RANDOM_CREATORS

from arc.arcdsl import dsl
from arc.arcdsl.arc_types import *


def filter_by_return_type(return_type: type = Grid) -> List[Callable]:
    """Filters functions according to if they return a specific type"""
    returners = []

    for name, func in inspect.getmembers(dsl):
        if inspect.isfunction(func):
            # Get the type hints for the function
            type_hints = get_type_hints(func)

            # Check if the return type matches the desired type
            if "return" in type_hints and type_hints["return"] == return_type:
                returners.append(
                    name
                )  # Append the function itself, not just the name

    return returners


def get_type_name(typ):
    """Helper function to get the name of a type, including generics"""
    if hasattr(typ, "__name__"):
        return typ.__name__
    elif hasattr(typ, "_name"):
        return typ._name
    elif hasattr(typ, "__origin__"):
        origin = typ.__origin__
        args = typ.__args__
        origin_name = get_type_name(origin)
        arg_names = [get_type_name(arg) for arg in args]
        return f"{origin_name}[{', '.join(arg_names)}]"
    else:
        return str(typ)


def get_input_parameters(fn: Callable) -> dict:
    """For a function, returns a dictionary of input parameters
    { parameter_name: parameter_type_name }
    """
    input_parameters = {}
    type_hints = get_type_hints(fn)

    for param, param_type in type_hints.items():
        if param != "return":
            input_parameters[param] = get_type_name(param_type)

    return input_parameters


def get_original_annotations(fn: Callable) -> Dict[str, Any]:
    """
    Retrieves the original type annotations for a function,
    preserving custom type aliases.
    """
    source = inspect.getsource(fn)
    signature_line = source.split("\n")[0].strip()
    params_str = signature_line[
        signature_line.index("(") + 1 : signature_line.rindex(")")
    ]
    params = [param.strip() for param in params_str.split(",") if ":" in param]

    annotations = {}
    for param in params:
        name, ann = param.split(":", 1)
        annotations[name.strip()] = ann.strip()

    return annotations


def prepare_kwargs(
    fn: Callable, grid: Union[np.array, List[List[int]]]
) -> dict:
    """Generate random parameters based on their original type annotations."""
    input_parameters = get_original_annotations(fn)
    kwargs = {}
    for k, v in input_parameters.items():
        if v == "Grid":
            kwargs[k] = grid
        elif v in RANDOM_CREATORS:
            kwargs[k] = RANDOM_CREATORS[v]()
        else:
            print(f"Could not get a value for {k} with type {v}")
    return kwargs
