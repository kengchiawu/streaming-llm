import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse

from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import parse_args, load

device = "cuda"
# print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    return past_key_values


def run_eval(args):
    model_name_or_path = args.model_name_or_path
    model,tokenizer = load(model_name_or_path)
    data = load_dataset(args.dataset_name, args.task, split=args.split)
    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    past_key_values = None
    num_eval_tokens = 0
    nll_sum = 0
    if args.enable_start_recent_kv_cache:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    os.makedirs(args.output_dir, exist_ok=True)
    f = open(f"{args.output_dir}/log.txt", "w")
    ff = open(f"{args.output_dir}/log_ppl.txt", "w")
    for text in data['text'][:data.num_rows]:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        input_ids = tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        pbar = tqdm(range(0,seq_len-1))
        for i in pbar:
            input_ids_i = input_ids[:,i:i+1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_i,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                target = input_ids[:,i+1:i+2].to(device).view(-1)
                loss = loss_fn(logits, target)
                
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            nlls.append(loss)
            pbar.set_description(f"nll: {loss.item():.2f},ppl:{torch.exp(loss).item():.2f}")
            nll_sum += loss.item()
            num_eval_tokens += 1
            print(f"nll: {loss.item():.2f},nll_sum: {nll_sum:.2f},num_eval_tokens: {num_eval_tokens}",file=f,flush=True)
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                args.num_eval_tokens = args.num_eval_tokens +50
                nll_sum = nll_sum / 50.0
                print(f"nll:{nll_sum:8.4f}  eval_tokens:{num_eval_tokens}",file=ff,flush=True)
                nll_sum = 0.0
        if args.num_eval_tokens is not None and num_eval_tokens >= 5000:
            break
    f.close()
    ff.close()
    ppl = torch.exp(torch.stack(nlls).mean())
    with open(f"{args.output_dir}/ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")

if __name__ == '__main__':
    args = parse_args()
    args.model_name_or_path = "C:/Users/y1116/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
    args.start_size = 4 
    args.recent_size = 1000
    #args.enable_start_recent_kv_cache = True
    #args.enable_pos_shift = True
# start_size=4 equals to attention sink =4, 
# especially when start_size=0, it equals to window attention.
# attention window = start_size + recent_size 
    args.num_eval_tokens = 50
    run_eval(args)


