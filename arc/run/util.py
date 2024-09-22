

def writing_llm_prompt(arc_problem_train_string, function_defs):
    prompt = f"""
        You are an AI tasked with solving Abstract Reasoning Corpus (ARC) problems. 
        You have access to a library of functions to help you solve these problems. 
        Your task is to propose a solution using these functions.

        Problem:
        {arc_problem_train_string}

        Available functions:
        {"".join(function_defs)}

        Propose a solution to transform the input into the output using the available functions. 
        Be specific about which functions you use and how you combine them. You do not need to
        import. Your solution should be a python function wrapped in ```. 
        If you need a function that's not available, you can write new ones.

        Your solution:
        """
    
    return prompt
