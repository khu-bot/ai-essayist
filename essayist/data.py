import json
from typing import Dict, List, TypedDict

import torch
from transformers import AutoTokenizer


class Datum(TypedDict):
    title: str
    content: str


def load_jsonl_data(path: str) -> List[Datum]:
    with open(path) as f:
        return [json.loads(l) for l in f]


class LanguageModelingDataset(torch.utils.data.Dataset):
    """LanguageModelingDataset

    Attributes:
        data: data for language modeling
        tokenizer: huggingface tokenizer
        max_length: token max length
    """

    def __init__(
        self, data: List[str], tokenizer: AutoTokenizer, max_length: int, use_token_type_ids: bool = True
    ) -> None:
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_token_type_ids = use_token_type_ids

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.data[index]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=self.use_token_type_ids,
        )
        inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]

        return inputs
