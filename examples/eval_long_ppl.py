import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

args = parse_args()
args.model_name_or_path = 'C:/Users/y1116/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549'
args.start_size = 4 
args.recent_size = 1000
args.enable_start_recent_kv_cache = True
args.enable_pos_shift = True
# start_size=4 equals to attention sink =4, 
# especially when start_size=0, it equals to window attention.
# attention window = start_size + recent_size 
args.num_eval_tokens = 50
data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None
#print(args.enable_start_recent_kv_cache)
if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    print("enable kv_cache")
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention
        print("enable llama pos shift")
        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )
        print("enable falcon pos shift")
        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )
        print("enable gpt-neox pos shift")
        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        print("mpt does not support pos shift")
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")
        print("disable pos shift")


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
ff = open(f"{args.output_dir}/log_ppl.txt", "w")
num_eval_tokens = 0
nll_sum = 0.0
for text in data["text"][: data.num_rows]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        nll_sum = nll_sum + neg_log_likelihood.item()
        num_eval_tokens += 1
        print(f'nll:{neg_log_likelihood.item():>10.4f}  nll_sum:{nll_sum:>10.4f} tokens_count:{num_eval_tokens}', file=f, flush=True)
        
        
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            args.num_eval_tokens = args.num_eval_tokens + 50
            nll_sum = nll_sum / 50.0
            print(f"nll:{nll_sum:8.4f}  eval_tokens:{num_eval_tokens}",file=ff,flush=True)
            nll_sum = 0.0

    if args.num_eval_tokens is not None and num_eval_tokens >= 5000:
        break

f.close()
ff.close()
ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
