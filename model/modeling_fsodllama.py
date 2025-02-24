import json, os
import torch
import transformers
from transformers import TrainerCallback
import time
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

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        image_path, caption, class_name, annotations, image_size = tuple([instance[key] for instance in instances]
                                  for key in ("image_path", "caption", "class_name", "annotations", "image_size"))
        
        batch = {
            "image_path": image_path,
            "caption": caption,
            "class_name": class_name,
            "annotations": annotations,
            "image_size": image_size
        }

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

        df = pd.read_json(data_path, lines=True)
        self.list_data_dict = df.to_dict(orient="records")
        self.tokenizer = tokenizer
        self.data_args = data_args
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        image_path = self.list_data_dict[idx]['image_path']
        caption = self.list_data_dict[idx]['caption']
        class_name = self.list_data_dict[idx]['class_name']
        annotations = self.list_data_dict[idx]['annotations']
        image_size = self.list_data_dict[idx]['image_size']

        data_dict = {
            "image_path" : image_path,
            "caption"    : caption,
            "class_name" : class_name,
            "annotations": annotations,
            "image_size" : image_size
        }

        return data_dict

import time
from transformers import TrainerCallback

class TimeRemainingCallback(TrainerCallback):
    def __init__(self, window_size=10):
        self.start_time = None
        self.total_steps = None
        self.step_times = []
        self.window_size = window_size
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_steps = state.max_steps
        self.last_step_time = self.start_time

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is None or self.total_steps is None:
            return

        current_time = time.time()
        completed_steps = state.global_step

        if completed_steps > 0:
            step_time = current_time - self.last_step_time  # 计算当前 step 的用时
            self.last_step_time = current_time  # 更新上一次 step 结束时间

            self.step_times.append(step_time)
            if len(self.step_times) > self.window_size:
                self.step_times.pop(0)

            avg_time_per_step = (
                sum(self.step_times) / len(self.step_times)
                if self.step_times else (current_time - self.start_time) / completed_steps
            )

            remaining_steps = self.total_steps - completed_steps
            remaining_time = remaining_steps * avg_time_per_step
            elapsed_time = current_time - self.start_time

            # 格式化时间（支持天）
            elapsed_time_formatted = self.format_time(elapsed_time)
            remaining_time_formatted = self.format_time(remaining_time)

            print(f"Elapsed time: {elapsed_time_formatted} | Estimated time remaining: {remaining_time_formatted}")

    @staticmethod
    def format_time(seconds):
        """格式化时间，支持天数"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if days > 0:
            return f"{days}days {hours:02}:{minutes:02}:{secs:02}"
        else:
            return f"{hours:02}:{minutes:02}:{secs:02}"