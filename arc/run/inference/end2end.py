import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc.arcdsl import PRIMITIVES
from arc.run.models import ARCResNetClassifier


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, help="ARC, REARC, or synthetic")
    parser.add_argument("--llm", default="llama-3.1-8b-instruct")
    args = parser.parse_args()

    # load resnet
    resnet = ARCResNetClassifier(len(PRIMITIVES))

    # load llm and tokenizer
    if args.llm == "llama-3.1-8b-instruct":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # load dataset

    return resnet, model


if __name__ == "__main__":
    main()
