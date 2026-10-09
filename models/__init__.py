from transformers import HfArgumentParser

from .arguments_live import LiveMemoryTrainingArguments
from .live_llama import build_live_llama as build_model_and_tokenizer
from .modeling_live import fast_greedy_generate, fast_greedy_generate_mem


def parse_args() -> LiveMemoryTrainingArguments:
    args, = HfArgumentParser(LiveMemoryTrainingArguments).parse_args_into_dataclasses()
    return args