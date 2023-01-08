import json
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import AutoTokenizer

from .utils import normalize_text


class Datum(TypedDict):
    title: str
    content: str
    summarizations: Optional[List[str]]


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
        self,
        data: List[Datum],
        tokenizer: AutoTokenizer,
        prompt_max_length: int,
        max_length: int,
        use_token_type_ids: bool = True,
    ) -> None:
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.use_token_type_ids = use_token_type_ids

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        datum = self.data[index]
        prompt, content = self.datum_to_string(datum)
        content = normalize_text(content)
        return self.create_tokenizer_inputs(prompt, content)

    @staticmethod
    def datum_to_string(datum: Datum) -> Tuple[str, str]:
        """Convert datum to string prompt, content"""
        prompt = f"제목: {datum['title']}\n"
        content = datum["content"]

        summarizations = datum.get("summarizations")
        if summarizations:
            summarization = " ".join(summarizations).replace("\n", " ")
            prompt = f"요약: {summarization}\n" + prompt
        return prompt, content

    def create_tokenizer_inputs(self, prompt: str, content: str) -> Dict[str, torch.Tensor]:
        self.tokenizer.truncation_side = "left"
        prompt_inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.prompt_max_length,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=self.use_token_type_ids,
        )
        prompt_inputs = {k: v.squeeze(dim=0) for k, v in prompt_inputs.items()}
        # `-100` is huggingface ignore index
        prompt_inputs["labels"] = torch.full_like(prompt_inputs["input_ids"], -100)

        self.tokenizer.truncation_side = "right"
        inputs = self.tokenizer(
            content,
            add_special_tokens=True,
            max_length=self.max_length - prompt_inputs["input_ids"].size(0),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=self.use_token_type_ids,
        )
        inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]

        for k in inputs:
            inputs[k] = torch.concat([prompt_inputs[k], inputs[k]], dim=0)

        return inputs
