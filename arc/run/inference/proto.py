import argparse
import os 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc.data import load_data, create_train_string
from arc.functions_library import list_all_function_defs
from arc.run import writing_llm_prompt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (faster, no logging)"
    )
    parser.add_argument(
        "--temperature", default=0.5, help="Temperature for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        default=500,
        help="Max new tokens during generation",
    )
    parser.add_argument(
        "--num_return_sequences",
        default=5,
        help="Number of outputs to generate in parallel per input"
    )
    args = parser.parse_args()

    # load train data
    train_problems, train_solutions = load_data("train")
    problem_ids = list(train_problems.keys())
    if args.debug:
        problem_ids = problem_ids[:5]

    # load model
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    function_defs = list_all_function_defs() # this will need to be modified so it's problem-specific

    # loop through the problems and get the train problems
    for pid in problem_ids:

        # makes a string of input 1: [[]], output 1: [[]] etc
        print("Creating train string")
        arc_problem_train_string = create_train_string(
            dataset=train_problems, problem_id=pid
        )

        # mix together into a nice prompt
        print("Formatting llm prompt")
        prompt = writing_llm_prompt(
            arc_problem_train_string=arc_problem_train_string,
            function_defs=function_defs,
        )

        # forward pass
        print("Tokenizing, forward pass ")
        input = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs_raw = model.generate(
            **input,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature
        )
        outputs_decoded = [
            tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs_raw
        ]

        # save the outputs for now, inspect + decide how to parse / execute
        savedir = os.path.join(os.environ['HOME'], 'arc', 'outputs', pid)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            for i, output in enumerate(outputs_decoded):
                with open(os.path.join(savedir, f"gen_{i+1}.txt"), 'w') as f:
                    f.write(output)

        # need to look for any new written functions, somehow decide whether to save
        # then execute function on the original train, see if they make provided solutions
        # finally execute on original test (could be a way off )


if __name__ == "__main__":
    main()
