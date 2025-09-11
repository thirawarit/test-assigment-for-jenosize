import copy
import json
import os
import argparse
import random 
import base64
import io
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import DataLoader, Dataset

from src.dataset.data_utils import (get_tokenize_qwen,
                                    pad_sequence)
from src.utils.constant import IGNORE_INDEX
from src.utils.params import DataArguments

class SupervisedDataset(Dataset):
    """A custom dataset class that inherits from `torch.utils.data.Dataset`."""

    def __init__(
        self,
        data_path: Union[str | List[str]],
        tokenizer: transformers.PreTrainedTokenizerFast,
        data_args: DataArguments,
        padding: bool = True,
    ):
        super(SupervisedDataset, self).__init__()

        if isinstance(data_path, str):
            if data_path.endswith(".json"):
                self.list_dict_data = json.load(open(data_path, "r"))
            elif data_path.endswith(".jsonl"):
                self.list_dict_data = open(data_path, "r").readlines()
            else:
                raise ValueError(f"data_path's extension should be '.json' or '.jsonl', but {os.path.basename(data_path)}.")
        elif isinstance(data_path, list):
            assert len(data_path) > 0, "len `data_path` == 0"
            if data_path[0].endswith(('.json', '.jsonl')):
                self.list_dict_data = []
                for p in data_path:
                    self.list_dict_data += open(p, 'r').readlines()
            else:
                self.list_dict_data = data_path

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.padding = padding


    def __len__(self):
        return len(self.list_dict_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.list_dict_data[idx]
        if not isinstance(data, dict):
            data = json.loads(data)

        if "metadata" in data and "settings" in data:
            data = data['metadata']
        
        # conversation
        sources = copy.deepcopy(get_tokenize_qwen(data["conversations"], self.tokenizer))
        all_input_ids = []
        all_labels_ids = []
        for _, idx in enumerate(range(0, len(sources), 2)):
            user_input = sources[idx]
            assistant_output = sources[idx + 1]
            
            encoding = self.tokenizer(user_input, padding=False, return_tensors='pt')
            prompt_input_ids = encoding["input_ids"]

            labels_encoding = self.tokenizer(assistant_output, add_special_tokens=False, padding=False, return_tensors='pt')
            labels_ids = labels_encoding["input_ids"]

            input_ids = torch.cat([prompt_input_ids, labels_ids], dim=1)
            labels = torch.cat(
                [
                    torch.full_like(prompt_input_ids, IGNORE_INDEX),
                    labels_ids,
                ],
                dim=1,
            )
            
            all_input_ids.append(input_ids.squeeze(0))
            all_labels_ids.append(labels.squeeze(0))

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels_ids, dim=0).to(torch.long)
        attention_mask = (input_ids > -100000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict
    
class DataCollatorForSupervisedDataset:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []

        for example in examples:
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        return data_dict
    
def make_supervised_data_module(tokenizer, data_args, model_id: str = None):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def main():
    data_path = []
    for p in args.data_path:
        data_path.extend(open(p, 'r').readlines())
    print('Total of JSON items:', len(data_path))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    
    dataset_args = DataArguments(data_path='mockup')
    dataset_args.media_dir = '/home/ubuntu/Datasets'
    print(args.data_path)
    dataset = SupervisedDataset(args.data_path, tokenizer, dataset_args)
    for _ in range(3):
        random_int = random.randint(0, len(data_path))
        print(f' example for test dataset at index: {random_int} '.center(60, '*'))
        print(dataset[random_int])
        print(' user + system '.center(40, '#'))
        print(tokenizer.decode(dataset[random_int]["input_ids"])[:100] + '...')
        print(' assistant '.center(40, '#'))
        print(tokenizer.decode(dataset[random_int]["labels"][dataset[random_int]["labels"]!=-100])[:100] + '...')
        print('*' * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python prepare_alldata.py',
        description='Test Prepare All Data',
    )
    parser.add_argument('--data-path', 
                      required=True, type=str, nargs='+',
                      help='a file\'s path for loading dataset')
    parser.add_argument('--model-id', 
                      required=False, default='Qwen/Qwen3-8B', type=str, nargs=1,
                      help='model id for loading dataset (default: %(default)s)')
    args = parser.parse_args()
    main()