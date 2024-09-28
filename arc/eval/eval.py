import json
import os
import re
from collections import defaultdict
from typing import Any, Callable


def extract_solution_from_llm_output(raw_text):
    try:
        llm_answer = raw_text.split(
            "return a List[List[int]].\n\n    Your solution:\n"
        )[1]
    except IndexError:
        import pdb

        pdb.set_trace()
    wrapped_split = llm_answer.split("```")
    for code in wrapped_split:
        if code.startswith("python"):
            code = code.split("python")[
                1
            ]  # edge case python is a variable in the function

        function_name, fn = extract_function(code)
        if not function_name:
            continue
        return code, function_name, fn

    return None, None, None


def extract_function(
    clean_function_string: str,
) -> tuple[str, Callable[..., Any]]:
    # add imports to the function
    import_statements = [
        "from typing import *",
        "from arc.arcdsl.dsl import *",
        "from arc.arcdsl.arc_types import *",
    ]
    clean_function_string = (
        "\n".join(import_statements) + "\n\n" + clean_function_string
    )

    # extract the string
    match = re.search(r"def\s+(\w+)", clean_function_string)
    if not match:
        return None, None

    function_name = match.group(1)
    namespace = {}
    try:
        exec(clean_function_string, namespace)
    except:
        return None, None

    return function_name, namespace[function_name]


def evaluate_output(
    raw_output, challenges, solutions, problem_id, verbose=True
):
    inputs = challenges[problem_id]["test"]
    expected_outputs = solutions[problem_id]

    correct = []

    fn_text, _, fn = extract_solution_from_llm_output(raw_output)
    if not fn_text:  # no executable function found in the llm output
        return correct

    if verbose:
        print(fn_text)

    for input, expected_output in zip(inputs, expected_outputs):
        try:
            obtained_output = fn(input)
            correct.append(int(obtained_output == expected_output))
        except:
            correct.append(0)

    return correct


def eval_all_files_in_folder(filepath, challenges, solutions, verbose=False):
    eval = defaultdict(dict)

    for results_file in os.listdir(filepath):
        correct = []

        with open(os.path.join(filepath, results_file), "r") as f:
            llm_submission = json.load(f)

        if "generations" not in llm_submission:
            continue  # making sure we are in the good type of file

        problem_id = results_file.strip(".json")

        for attempt in llm_submission["generations"]:
            correct.append(
                tuple(
                    evaluate_output(
                        attempt,
                        challenges,
                        solutions,
                        problem_id,
                        verbose=verbose,
                    )
                )
            )

        eval["n_correct_per_attempt"][problem_id] = correct

    return eval
