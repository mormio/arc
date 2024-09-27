def writing_llm_prompt(arc_problem_train_string, function_defs):
    prompt = f"""
    You are tasked wiht solving abstract reasoning problems.
    You have access to a library of functions to help you solve these problems.
    Your task is to propose a solution using these functions and any other new
    functions you can choose to write. You may not need to use any of the functions
    in the provided library.

    Problem:
    {arc_problem_train_string}

    Available functions:
    {"".join(function_defs)}

    Propose a solution to transform the input into the output.
    Be specific about which functions you use and how you combine them. You do not need to
    import. Your solution should be a python function wrapped in ```.

    Your solution:
    """

    return prompt


def make_prompt_with_recommendations(
    arc_problem_train_string, primitives_shortlist
):
    prompt = f"""
    You are tasked wiht solving abstract reasoning problems. Your helper has recommended
    a selection of primitive functions to help you solve a particular problem. You must
    propose a solution using these functions and any other novel functions you may choose to write.
    You may not need to use all or any of the recommended functions.

    Problem:
    {arc_problem_train_string}

    Recommended function primitives:
    {"".join(primitives_shortlist)}

    Propose a solution to transform the problem's input into the output.
    Be specific about which functions you use and how you combine them. You do not need to import.
    Your solution should be a python function wrapped in ```.

    Your solution:
    """

    return prompt


def filter_by_binary(items, binary_filter):
    """For example, good for getting the functions predicted by the resnet."""
    if len(items) != len(binary_filter):
        raise ValueError("Lists must have the same length")

    return [item for item, keep in zip(items, binary_filter) if keep]


def aggregate_problem_predictions(preds, method=any):
    """Aggregates across all the predictions for the examples belonging to a single problem."""

    assert method in [any, all], "aggregation must be either any or or"
    for pid, pred_list in preds.items():
        agg = [
            method(row[i] for row in pred_list)
            for i in range(len(pred_list[0]))
        ]
        preds[pid] = agg

    return preds
