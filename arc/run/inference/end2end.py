import argparse
import gc
import inspect
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from arc.arcdsl import PRIMITIVES
from arc.arcdsl import dsl as dsl_mod
from arc.arcdsl import solvers as solvers_mod
from arc.data import (
    ARCDataLoader,
    ARCDataset,
    get_primitives_vector_for_problem,
    load_data,
)
from arc.eval import eval_all_files_in_folder, extract_solution_from_llm_output
from arc.run import (
    LLM,
    LLM_MODELS,
    aggregate_problem_predictions,
    extract_code_from_text,
    extract_reasoning_from_text,
    filter_by_binary,
)
from arc.run.prompting import make_system_prompt, make_user_prompt
from arc.run.resnet import ARCResNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # prep llm
    print("Loading LLM.")
    llm = LLM(model_name=args.llm)

    print("Starting LLM forward pass...")
    llm_results_savedir = forward_pass_llm(
        llm,
        args,
        preds,
        train_problems,
    )

    print(
        "LLM forward passes complete. Starting evaluations and analysis for problems."
    )

    # evaluate
    eval_results = eval_all_files_in_folder(
        llm_results_savedir, train_problems, train_sols, False
    )
    # save eval
    eval_path = os.path.join(llm_results_savedir, "evaluations.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saving evaluation results to: {eval_path}")

    # analyse number of primitives recommended vs used
    analysis_results = analyze_prompt_usage(llm_results_savedir)
    # save analysis
    analysis_path = os.path.join(llm_results_savedir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"Saving analysis results to: {analysis_path}")

    # ========================== FINISHED ==========================


def analyze_prompt_usage(path2results):
    analysis = defaultdict(dict)

    for results_file in os.listdir(path2results):
        with open(os.path.join(path2results, results_file), "r") as f:
            llm_submission = json.load(f)

        if "generations" not in llm_submission:
            continue  # making sure we are in the good type of file

        problem_id = results_file.strip(".json")

        # find the primitives that would have been recommended in the prompt
        recommended_primitives_vector = llm_submission[
            "recommended_primitives_vector"
        ]
        recommended_primitives = filter_by_binary(
            PRIMITIVES, recommended_primitives_vector
        )

        # loop through saved llm attempts
        n_primitives_used = []
        intersection = []
        for attempt in llm_submission["generations"]:
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
        analysis["n_primitives_recommended"][problem_id] = len(
            recommended_primitives
        )
        analysis["n_primitives_used"][problem_id] = n_primitives_used
        analysis["intersection"][problem_id] = intersection

    return analysis


def forward_pass_llm(
    llm,
    args,
    resnet_preds,
    train_problems,
):
    print(f"Starting to generate answers for {len(train_problems)} problems.")

    for problem_id in tqdm(
        list(train_problems.keys()), desc="Problems in LLM"
    ):
        torch.cuda.empty_cache()

        # ground truth primitives if available
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

        # get source code for recommended primitives
        primitives_shortlist = [
            inspect.getsource(getattr(dsl_mod, p)) + "\n\n"
            for p in (
                gt_primitives
                if args.recommend_gt_primitives
                else primitives_shortlist
            )
        ]

        # make system and user prompts
        system_prompt = make_system_prompt()
        user_prompt = make_user_prompt(
            train_problems[problem_id]["train"],
            args.llm,
            primitives_shortlist,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # forward pass
        outputs_decoded = llm.generate(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            return_list_of_strings=True,
        )

        # extract reasoning
        reasoning = [extract_reasoning_from_text(t) for t in outputs_decoded]
        hypotheses = [extract_code_from_text(t) for t in outputs_decoded]

        # save
        save_results = {
            "raw_generations": outputs_decoded,
            "reasoning": reasoning,
            "hypotheses": hypotheses,
            "recommended_primitives_vector": (
                gt_primitives_label
                if args.recommend_gt_primitives
                else resnet_preds[problem_id]
            ),
        }

        # save the outputs
        if args.savedir is None:
            savedir = os.path.join(
                os.environ["HOME"], "arc", "outputs", args.llm
            )
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
        "--dataset",
        type=str,
        default="ARC",
        help="ARC, REARC, or synthetic",
    )
    parser.add_argument("--llm", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument(
        "--split", type=str, default="train", help="Data split."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="For llm sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="For LLM sampling.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="Number of samples per prompt in the LLM.",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="Only do the easy ARC subset.",
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
    parser.add_argument(
        "--savedir",
        type=str,
        help="Where to save llm generations, evaluations, and analysis",
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Optional string used by runner.py for making savedir",
    )
    args = parser.parse_args()

    assert (
        args.llm in LLM_MODELS.keys()
    ), f"llm arg must be one of {LLM_MODELS.keys()}"
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
