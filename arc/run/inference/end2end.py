import argparse
import gc
import inspect
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc.arcdsl import PRIMITIVES
from arc.arcdsl import dsl as dsl_mod
from arc.arcdsl import solvers as solvers_mod
from arc.data import (
    ARCDataLoader,
    ARCDataset,
    create_train_string,
    get_primitives_vector_for_problem,
    load_data,
)
from arc.eval import evaluate_output, extract_solution_from_llm_output
from arc.run import (
    aggregate_problem_predictions,
    filter_by_binary,
    make_prompt_with_recommendations,
)
from arc.run.models import ARCResNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {"llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"}


def main():
    # args
    args = get_arguments()
    print("Args parsed.")

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    # load resnet
    print("Loading resnet...")
    resnet = ARCResNetClassifier(len(PRIMITIVES)).to(DEVICE)
    print("Resnet loaded.")

    # resnet dataset and dataloader
    print(f"Creating dataset with {' easy' if args.easy else ''} problems")
    dataset = ARCDataset(split=args.split, debug=args.debug, easy=args.easy)
    resnet_dataloader = ARCDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count()),
        normalize=True,
    )
    problem_ids = [x[-1] for x in dataset.data]
    print(
        "Dataset and dataloader for resnet created. Starting resnet forward passes."
    )

    # forward pass the resnet
    preds = forward_pass_resnet(
        resnet=resnet,
        dataloader=resnet_dataloader,
        device=DEVICE,
    )

    print("Resnet forward passes complete. Aggregating per problem...")
    # aggregate the results, since there will be ~1000 passes per problem for REARC dataset
    preds = aggregate_problem_predictions(preds)

    # delete the resnet and resnet dataloader
    del resnet
    del resnet_dataloader
    torch.cuda.empty_cache()

    if args.dataset != "ARC":
        print(
            "Unsure how the create_train_string function will behave on non-ARC dataset.."
        )
    train_problems, train_sols = load_data(split="train")
    train_problems = {
        k: v for k, v in train_problems.items() if k in problem_ids
    }

    # load llm and tokenizer
    model_name = MODELS.get(args.llm)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.bos_token_id
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Starting LLM forward pass...")
    llm_results_savedir = forward_pass_llm(
        llm,
        tokenizer,
        args,
        preds,
        train_problems,
    )

    print("LLM forward passes complete. Starting evaluations for problems.")
    # evaluate
    assert (
        len(os.listdir(llm_results_savedir)) == len(train_problems) + 1
    )  # +1 for the evaluations file
    eval_results = eval_llm_attempts(
        llm_results_savedir, train_problems, train_sols, problem_ids
    )
    # save eval
    eval_path = os.path.join(llm_results_savedir, "evaluations.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saving evaluation results to: {eval_path}")

    # analyse number of primitives recommended vs used
    analysis_results = analyze_prompt_usage(llm_results_savedir, problem_ids)
    # save analysis
    analysis_path = os.path.join(llm_results_savedir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"Saving analysis results to: {analysis_path}")

    # ========================== FINISHED ==========================


def analyze_prompt_usage(path2results, problem_ids):
    analysis = defaultdict(dict)

    for problem in problem_ids:
        full_path = os.path.join(path2results, problem + ".json")
        with open(full_path, "r") as f:
            attempts = json.load(f)

        # find the primitives that would have been recommended in the prompt
        recommended_primitives_vector = attempts[
            "recommended_primitives_vector"
        ]
        recommended_primitives = filter_by_binary(
            PRIMITIVES, recommended_primitives_vector
        )

        # loop through saved llm attempts
        n_primitives_used = []
        intersection = []
        for key, attempt in attempts.items():
            if key == "recommended_primitives_vector":
                continue
            clean_function = extract_solution_from_llm_output(attempt)
            # check how many primitives from the dsl are used
            n_primitives_used.append(
                len([x for x in PRIMITIVES if x in clean_function])
            )
            # check how many of these were also recommended
            intersection.append(
                len(
                    [
                        x
                        for x in PRIMITIVES
                        if (x in clean_function)
                        and (x in recommended_primitives)
                    ]
                )
            )

        # store before moving onto next problem
        analysis["n_primitives_recommended"][problem] = len(
            recommended_primitives
        )
        analysis["n_primitives_used"][problem] = n_primitives_used
        analysis["intersection"][problem] = intersection

    return analysis


def eval_llm_attempts(path2results, challenges, solutions, problem_ids):
    problems_solved = [0] * len(problem_ids)
    problems_partially_solved = [0] * len(problem_ids)
    total_out_of = 0
    total_correct = 0

    for i, problem in enumerate(problem_ids):
        with open(os.path.join(path2results, problem + ".json"), "r") as f:
            problem_hypotheses = json.load(f)

        # evaluate
        for key, hypothesis in problem_hypotheses.items():
            if key == "recommended_primitives_vector":
                continue
            correct, out_of = evaluate_output(
                raw_output=hypothesis,
                challenges=challenges,
                solutions=solutions,
                problem_id=problem,
            )
            if correct == out_of:
                problems_solved[i] = 1
            if correct > 0 and correct < out_of:
                problems_partially_solved[i] = 1
            total_out_of += out_of
            total_correct += correct

    return {
        "problems_solved": problems_solved,
        "problems_partially_solved": problems_partially_solved,
        "total_out_of": total_out_of,
        "total_correct": total_correct,
    }


def forward_pass_llm(
    llm,
    tokenizer,
    args,
    resnet_preds,
    train_problems,
    savedir=None,
):
    print(f"There are {len(train_problems)} problems.")

    llm.eval()
    for problem_id in tqdm(
        list(train_problems.keys()), desc="Problems in LLM"
    ):
        torch.cuda.empty_cache()
        # get the problem id
        gt_primitives_label = get_primitives_vector_for_problem(
            problem_id, solvers_mod
        )
        gt_primitives = (
            filter_by_binary(PRIMITIVES, gt_primitives_label)
            if gt_primitives_label
            else "not known"
        )

        # get the resnet's primitive predictions
        primitives_shortlist = filter_by_binary(
            PRIMITIVES, resnet_preds[problem_id]
        )

        print(
            f"""For problem {problem_id} the primitives shortlist is
            {primitives_shortlist} and the actual primitives are {gt_primitives}. """
        )

        # get source code for recommended primitives
        primitives_shortlist = [
            inspect.getsource(getattr(dsl_mod, p)) + "\n\n"
            for p in (
                gt_primitives
                if args.recommend_gt_primitives
                else primitives_shortlist
            )
        ]

        # Make the prompt
        problem_train_string = create_train_string(
            dataset=train_problems, problem_id=problem_id
        )
        prompt = make_prompt_with_recommendations(
            problem_train_string, primitives_shortlist
        )

        # sample
        input = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs_raw = llm.generate(
            **input,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
        )
        outputs_decoded = [
            tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs_raw
        ]

        save_results = {x: out for x, out in enumerate(outputs_decoded)}
        save_results["recommended_primitives_vector"] = (
            gt_primitives_label
            if args.recommend_gt_primitives
            else resnet_preds[problem_id]
        )

        # save the outputs for now, inspect + decide how to parse / execute
        if savedir is None:
            savedir = os.path.join(os.environ["HOME"], "arc", "outputs")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        problem_savedir = os.path.join(savedir, problem_id + ".json")
        with open(problem_savedir, "w") as f:
            json.dump(save_results, f)

    return savedir


def forward_pass_resnet(resnet, dataloader, device, thresh=0.5):
    resnet.eval()

    problemwise_predictions = defaultdict(list)
    for batch in dataloader:
        inputs = batch["combined_input"].to(device)
        problems = batch["problem_ids"]

        with torch.no_grad():
            outputs = resnet(inputs)

        preds = (outputs > thresh).float()

        for pid, pred in zip(problems, preds):
            problemwise_predictions[pid].append(pred)

    return problemwise_predictions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="ARC", help="ARC, REARC, or synthetic"
    )
    parser.add_argument("--llm", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument(
        "--split", type=str, default="train", help="Data split."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_new_tokens", type=int, default=300, help="For llm sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="For LLM sampling."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="Number of samples per prompt in the LLM.",
    )
    parser.add_argument(
        "--easy", action="store_true", help="Only do the easy ARC subset."
    )
    parser.add_argument(
        "--resnet_thresh",
        type=float,
        default=0.5,
        help="Threshold for sigmoid on resnet outputs",
    )
    parser.add_argument(
        "--recommend_gt_primitives",
        action="store_true",
        help="Where available, provide the true primitives used in the REARC solvers, in the llm prompt.",
    )
    parser.add_argument(
        "--pred_aggregation",
        default="all",
        help="any or or, to aggregate predictions over a single problem's samples",
    )
    args = parser.parse_args()

    assert args.llm in MODELS.keys(), f"llm arg must be one of {MODELS.keys()}"
    assert args.pred_aggregation in [
        "any",
        "all",
    ], "pred aggregation arg must be one of any (or) or all (and)"
    if args.pred_aggregation == "any":
        args.pred_aggregation == any
    elif args.pred_aggregation == "all":
        args.pred_aggregation == all

    return args


if __name__ == "__main__":
    main()
