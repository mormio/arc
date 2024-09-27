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


def extract_functions(text) -> List[str]:
    """extract functions from llm output text"""
    function_pattern = r'def\s+\w+\s*\([^)]*\)\s*->\s*List\[List\[int\]]:\s*(?:"""(?:.*?\n)*?.*?""")?\s*(?:.*?\n)*?.*?\n\n'

    # Find all matches in the text
    functions = re.findall(function_pattern, text, re.DOTALL)

    # Strip leading/trailing whitespace from each function
    functions = [
        func.strip() for func in functions if "return" in func.strip()
    ]

    return functions
