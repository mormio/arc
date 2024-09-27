import re 
from typing import List 


def get_def_and_docstring(function_string: str) -> str:
    """extract function def + docstring as a summary for the llm to use it."""
    if '"""' in function_string:
        sig = '"""'.join(function_string.split('"""')[:2]) + '"""' + "\n\n"
    elif "'''" in function_string:
        sig = "'''".join(function_string.split("'''")[:2]) + "'''" + "\n\n"
    else:
        raise ValueError(
            f"No docstring found in this function: {function_string}"
        )
    return sig
