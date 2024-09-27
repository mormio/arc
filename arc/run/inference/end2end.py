import argparse
import inspect
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc.arcdsl import PRIMITIVES
from arc.arcdsl import dsl as dsl_mod
from arc.arcdsl import get_def_and_docstring
from arc.data import EASY_SUBSET, ARCDataLoader, ARCDataset
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

    # load resnet
    resnet = ARCResNetClassifier(len(PRIMITIVES)).to(DEVICE)

    # load dataset, dataloader
    if args.easy:
        problem_ids = EASY_SUBSET
        if args.debug:
            problem_ids = problem_ids[:5]
    else:
        problem_ids = None

    # resnet dataset and dataloader
    dataset = ARCDataset(split=args.split, debug=args.debug, easy=args.easy)
    resnet_dataloader = ARCDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count()),
        normalize=True,
    )

    # forward pass the resnet
    preds = forward_pass_resnet(
        resnet=resnet,
        dataloader=resnet_dataloader,
        device=DEVICE,
    )

    # aggregate the results, since there will be ~1000 passes per problem for REARC dataset
    preds = aggregate_problem_predictions(preds)

    # delete the resnet and resnet dataloader
    del resnet
    del resnet_dataloader

    # make the llm dataloader
    llm_dataloader = ARCDataLoader(
        dataset=dataset,
        batch_size=1,  # we want to sample many per problem
        num_workers=min(4, os.cpu_count()),
    )

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
    forward_pass_llm(llm, tokenizer, llm_dataloader, args, preds)


def forward_pass_llm(
    llm, tokenizer, dataloader, args, primitives_dict, savedir=None
):
    if not savedir:
        savedir = os.path.join(os.environ["HOME"], "arc", "outputs")

    llm.eval()
    for batch in tqdm(dataloader, desc="Problems in LLM"):
        # get the problem id
        problem_id = batch["problem_ids"][0]
        label = batch["labels"][0]
        actual_primitives = filter_by_binary(PRIMITIVES, label)

        # get the resnet's primitive predictions
        resnet_preds = primitives_dict[problem_id]
        primitives_shortlist = filter_by_binary(PRIMITIVES, resnet_preds)

        print(
            f"""For problem {problem_id} the primitives shortlist is
            {primitives_shortlist} and the actual primitives are {actual_primitives}. """
        )

        # extract the function definition + docstring for each recommendation
        primitives_shortlist = [
            get_def_and_docstring(inspect.getsource(getattr(dsl_mod, p)))
            + "\n\n"
            for p in primitives_shortlist
        ]

        # Make the prompt
        prompt = make_prompt_with_recommendations(
            problem_id, primitives_shortlist
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

        # save the outputs for now, inspect + decide how to parse / execute
        problem_savedir = os.path.join(savedir, problem_id)
        if not os.path.exists(problem_savedir):
            os.makedirs(problem_savedir)
            with open(problem_savedir, "w") as f:
                json.dump(save_results, f)


def forward_pass_resnet(resnet, dataloader, device, thresh=0.5):
    resnet.eval()

    preds = defaultdict(list)
    for batch in dataloader:
        inputs = batch["combined_input"].to(device)
        problems = batch["problem_ids"]

        with torch.no_grad():
            outputs = resnet(inputs)

        preds = (outputs > thresh).float()

        for pid, pred in zip(problems, preds):
            print("pred: ", pred)
            preds[pid].extend(pred)

    return preds


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, help="ARC, REARC, or synthetic")
    parser.add_argument("--llm", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument(
        "--split", type=str, default="train", help="Data split."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_new_tokens", type=int, default=600, help="For llm sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="For LLM sampling."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=64,
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
        "--pred_aggregation",
        default="any",
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
