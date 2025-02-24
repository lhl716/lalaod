import json, os
import torch
import transformers
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from data.util import load_data_from_batches,load_data_for_eval
from typing import Dict, Optional, Sequence

IGNORE_INDEX = -100
local_rank = None
def get_list_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_list_shape(lst[0]) if lst else [0]
    return []

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def get_json_data(data_path):
    list_data_dict = []
        
        # 逐行读取 JSONL 文件
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            list_data_dict.append({
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.long).to('cuda'),
                "attention_mask": torch.tensor(data["attention_mask"], dtype=torch.long).to('cuda'),
                "labels": torch.tensor(data["labels"], dtype=torch.long).to('cuda'),
                "sup_visual_tokens": data["sup_visual_tokens"],
                "que_visual_tokens": data["que_visual_tokens"]
            })
    return list_data_dict

def load_csv_to_list(data_path):
    list_data_dict = []
    dataframe = pd.read_csv(data_path)
    
    for _, row in dataframe.iterrows():
        list_data_dict.append({
            "input_ids": torch.tensor(eval(row["input_ids"]), dtype=torch.long).to('cuda'),
            "attention_mask": torch.tensor(eval(row["attention_mask"]), dtype=torch.long).to('cuda'),
            "labels": torch.tensor(eval(row["labels"]), dtype=torch.long).to('cuda'),
            "sup_visual_tokens": eval(row["sup_visual_tokens"]),
            "que_visual_tokens": eval(row["que_visual_tokens"])
        })
    
    return list_data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, sup_visual_tokens, que_visual_tokens = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "attention_mask", "sup_visual_tokens", "que_visual_tokens"))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            sup_visual_tokens=sup_visual_tokens,
            que_visual_tokens=que_visual_tokens
        )

        return batch

class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

class PrepareDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(PrepareDataset).__init__()
        
        print(f"Formatting Datasets from {data_path}...")

        self.list_data_dict = load_data_from_batches(data_path)
        
        self.tokenizer = tokenizer
        self.data_args = data_args
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = self.list_data_dict[idx]['input_ids']
        attention_mask = self.list_data_dict[idx]['attention_mask']
        labels = self.list_data_dict[idx]['labels']
        sup_visual_tokens = self.list_data_dict[idx]['sup_visual_tokens']
        que_visual_tokens = self.list_data_dict[idx]['que_visual_tokens']
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        labels = labels.squeeze()
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sup_visual_tokens': sup_visual_tokens,
            'que_visual_tokens': que_visual_tokens
        }
        return data_dict
