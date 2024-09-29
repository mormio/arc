import argparse
import json
import os
from abc import ABC
from typing import List

import torch
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

LLM_MODELS = {
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gpt-4o": "gpt-4o",
    "gpt-4o-gs": "gpt-4o-gs",
}


class LLM(ABC):
    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name

        # open ai
        if self.model_name in ["gpt-4o-gs", "gpt-4o"]:
            self.azure_endpoint = os.environ.get("OPENAI_GPT4O_API_BASE")
            self.api_key = os.environ.get("OPENAI_GPT4O_API_KEY")
            self.api_version = os.environ.get("OPENAI_API_VERSION")
            self.model = ARCAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                model_name=self.model_name,
            )
            self.generate = self.model.generate

        # HF automodel for causal LM gang
        elif self.model_name in ["llama-3.1-8b-instruct"]:
            self.model = ARCPipeline(
                LLM_MODELS[self.model_name], device_map="auto"
            )
            self.generate = self.model.generate
        else:
            raise ValueError(
                f"""Model name not recognised, got {model_name} but
                must be one of {list(LLM_MODELS.keys())}"""
            )


class ARCAzureOpenAI(AzureOpenAI):
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        model_name: str,
    ):
        super().__init__(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.model_name = model_name

    def estimate_tokens(self, messages: List[dict]) -> int:
        # This is a rough estimation. Actual token count may vary.
        token_estimate = 0
        for message in messages:
            # Estimate 4 tokens for message format
            token_estimate += 4
            for key, value in message.items():
                # Rough estimate: 1 token per 4 characters
                token_estimate += len(json.dumps(value)) // 4
        # Add 2 tokens for assistant reply format
        token_estimate += 2
        return token_estimate

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int,
        num_return_sequences: int,
        temperature: float,
        return_list_of_strings: bool = False,
    ):
        tokens_in_messages = self.estimate_tokens(messages)
        max_tokens = tokens_in_messages + max_new_tokens

        response = self.chat.completions.create(
            model=self.model_name,
            response_format={"type": "text"},
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_return_sequences,
        )

        generated_texts_list_of_strings = [
            choice.message.content for choice in response.choices
        ]

        if return_list_of_strings:
            return generated_texts_list_of_strings
        else:
            return response.choices


class ARCPipeline:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        model_kwargs: dict = {"torch_dtype": torch.bfloat16},
        return_full_text: bool = False,
    ):
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs=model_kwargs,
            device_map=device_map,
            return_full_text=return_full_text,
        )

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int,
        num_return_sequences: int,
        temperature: float,
        return_list_of_strings: bool = False,
        **pipeline_kwargs,
    ):
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            **pipeline_kwargs,
        )
        if return_list_of_strings:
            return [x["generated_text"] for x in outputs]
        else:
            return outputs


class ARCAutoModelForCausalLM(AutoModelForCausalLM):
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(
            model_name,
            device_map,
            torch_dtype,
        )
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.device = next(self.parameters()).device

    def generate_responses(self, prompt_str: str, args: argparse.Namespace):
        input = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        outputs_raw = self.generate(
            **input,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
        )
        outputs_decoded = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs_raw
        ]
        return outputs_decoded
