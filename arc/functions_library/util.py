import re 
import os 
import inspect 
from typing import List
from arc.functions_library import functions as fn_

def get_def_and_docstring(function_string: str) -> str:
    """ extract function def + docstring as a summary for the llm to use it. """
    if '"""' in function_string:
        sig = '"""'.join(function_string.split('"""')[:2]) + '"""' + "\n\n"
    elif "'''" in function_string:
        sig = "'''".join(function_string.split("'''")[:2]) + "'''" + "\n\n"
    else:
        raise ValueError(f"No docstring found in this function: {function_string}")
    return sig

def extract_functions(text) -> List[str]:
    """ extract functions from llm output text"""
    function_pattern = r'def\s+\w+\s*\([^)]*\)\s*->\s*List\[List\[int\]]:\s*(?:"""(?:.*?\n)*?.*?""")?\s*(?:.*?\n)*?.*?\n\n'
    
    # Find all matches in the text
    functions = re.findall(function_pattern, text, re.DOTALL)
    
    # Strip leading/trailing whitespace from each function
    functions = [func.strip() for func in functions if 'return' in func.strip()]
    
    return functions

def extract_imports(text: str) -> List[str]:
    """Extract import lines from llm output text """
    import_pattern = r'^(import\s+.+|from\s+.+\s+import\s+.+)$'
    
    # Find all matches in the text
    imports = re.findall(import_pattern, text, re.MULTILINE)
    
    return imports

def update_file(functions_list: List[str], imports_list: List[str]):
    """ Add new imports and functions to the functions library. 
    """

    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, 'functions.py')

    # find existing functions 
    with open(file, 'r') as f:
        file_content = f.read()
        existing_functions = extract_functions(file_content)
        existing_imports = extract_imports(file_content)
    
    new_functions = [f for f in functions_list if f not in existing_functions]
    new_imports = [i for i in imports_list if i not in existing_imports]

    with open(file, 'a') as f:
        for im in new_imports:
            f.write(im + "\n\n")
        for func in new_functions:
            f.write(func + "\n\n")

def list_all_function_defs(fpath:str = None) -> List[str]:
    """ loads the functions in the file and returns a list of their definitions + docstrings. 
    """
    if not fpath:
        fpath = inspect.getfile(fn_)

    with open(fpath, 'r') as f:
        functions_content = f.read()

    functions = extract_functions(functions_content)
    defs = [get_def_and_docstring(f) for f in functions]

    return defs 