
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset


dataset_name = "wikitext"
task = "wikitext-2-raw-v1"
data = load_dataset(dataset_name, task ,split="test")
i=0
for text in data["text"][: data.num_rows]:
    i=i+1
print(i, data.num_rows)