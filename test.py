
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import ssl
import urllib.request
import json

dataset_name = "wikitext"
task = "wikitext-2-raw-v1"
data = load_dataset(dataset_name, task ,split="test")

model_name_or_path = "lmsys/vicuna-13b-v1.3"
print(f"Loading model from {model_name_or_path} ...")
tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
            tokenizer.pad_token_id = 0

model.eval()