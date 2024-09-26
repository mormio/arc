def writing_llm_prompt(arc_problem_train_string, function_defs):
    prompt = f"""
    You are an AI tasked with solving Abstract Reasoning Corpus (ARC) problems.
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
