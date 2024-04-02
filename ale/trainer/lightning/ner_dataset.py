from pathlib import Path
from typing import List, Tuple, Callable

import srsly
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from ale.trainer.lightning.utils import derive_labels


class AleNerDataModule(LightningDataModule):
    def __init__(self, data_dir: str = None, model_name: str = None, labels: List[str] = None, batch_size: int = 32,
                 num_workers: int = 1, train_filter_func: Callable = lambda x: x):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2label, self.label2id, _ = derive_labels(labels)
        self.num_workers = num_workers
        self.train = None
        self.dev = None
        self.test = None
        self.train_filter_func = train_filter_func

    def prepare_data(self):
        self.train = self.load_dataset(self.data_dir / "train.jsonl")
        self.dev = self.load_dataset(self.data_dir / "dev.jsonl")
        self.test = self.load_dataset(self.data_dir / "test.jsonl")

    def load_dataset(self, path: Path):
        result = []
        for entry in srsly.read_jsonl(path):
            text = entry["text"]
            labels = entry["labels"]

            tokenized = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=True)
            token_labels, tokens_text = self.create_token_labels(labels, tokenized)
            token_labels = [self.label2id[label] for label in token_labels]

            if "id" in entry:
                result.append({"tokens": tokenized, "labels": token_labels, "text": text,
                               "token_text": tokens_text, "id": entry["id"]})
            else:
                result.append({"tokens": tokenized, "labels": token_labels, "text": text,
                               "token_text": tokens_text})

        return result

    def create_token_labels(self, labels, tokenized):
        tokens_text = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        cleaned_tokens = [token.lstrip("Ġ") for token in tokens_text]
        token_labels = self.char_to_token_labels(tokenized, labels)
        return token_labels, cleaned_tokens

    def collate(self, batch, pad_label_value: int = -100):
        # Get individual elements from the batch
        max_length = max(len(example['tokens']['input_ids']) for example in batch)  # Determine the max length

        # Initialize lists to hold the batch data
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_offset_mapping = []

        for data in batch:
            input_ids = data['tokens']['input_ids']
            attention_mask = data['tokens']['attention_mask']
            labels = data['labels']
            offsets = data['tokens']['offset_mapping']

            padding_length = max_length - len(input_ids)

            # Pad the 'input_ids', 'attention_mask', and 'labels'
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([pad_label_value] * padding_length)

            # Append the padded data to the batch lists
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_offset_mapping.append(offsets)

        # Convert lists to PyTorch tensors
        batch_data = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'text': [entry["text"] for entry in batch],
            'token_text': [entry["token_text"] for entry in batch],
            'offset_mapping': batch_offset_mapping
        }

        return batch_data

    def train_dataloader(self):
        filtered = self.train_filter_func(self.train)
        return DataLoader(filtered, batch_size=self.batch_size, collate_fn=self.collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dev, batch_size=self.batch_size, collate_fn=self.collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass  # not used

    def teardown(self, stage: str):
        pass  # Used to clean-up when the run is finished

    def char_to_token_labels(self, batch_encoding: BatchEncoding, char_labels: List[Tuple[int, int, str]]):
        token_labels = ["O"] * len(batch_encoding["offset_mapping"])

        for start_char, end_char, label in char_labels:
            # Flag to denote the first token in the span
            first_token = True
            for i, (start, end) in enumerate(batch_encoding["offset_mapping"]):
                # Skip tokens that are not part of the original text (special tokens)
                if start == end == 0:
                    continue

                # Check if the current token overlaps with the current annotation span
                if start < end_char and end > start_char:
                    if first_token:
                        token_labels[i] = "B-" + label
                        first_token = False
                    else:
                        token_labels[i] = "I-" + label

        return token_labels


class PredictionDataModule(LightningDataModule):
    def __init__(self, texts: List[str] = None, model_name: str = None,
                 batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prediction_set = self.process_texts(texts)

    def process_texts(self, texts: List[str]):
        result = []
        for text in texts:
            tokenized = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True)
            tokens_text = self.create_token_labels(text)

            result.append({"tokens": tokenized, "text": text, "token_text": tokens_text})

        return result

    def create_token_labels(self, text):
        tokens_text = self.tokenizer.tokenize(text)
        cleaned_tokens = [token.lstrip("Ġ") for token in tokens_text]
        return cleaned_tokens

    def collate(self, batch):
        """
        Collates a batch of data samples for training or evaluation, performing padding.

        Args:
            batch (list): A list of dictionaries containing tokenized data and labels.

        Returns:
            dict: A dictionary with batched tensors for 'input_ids', 'attention_mask',
                  and 'labels'.
        """

        # Get individual elements from the batch
        max_length = max(len(example['tokens']['input_ids']) for example in batch)  # Determine the max length

        # Initialize lists to hold the batch data
        batch_input_ids = []
        batch_attention_mask = []
        batch_offset_mapping = []

        for data in batch:
            input_ids = data['tokens']['input_ids']
            attention_mask = data['tokens']['attention_mask']
            offsets = data['tokens']['offset_mapping']

            padding_length = max_length - len(input_ids)

            # Pad the 'input_ids', 'attention_mask', and 'labels'
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)

            # Append the padded data to the batch lists
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_offset_mapping.append(offsets)

        batch_data = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'text': [entry["text"] for entry in batch],
            'token_text': [entry["token_text"] for entry in batch],
            'offset_mapping': batch_offset_mapping
        }

        return batch_data

    def predict_dataloader(self):
        return DataLoader(self.prediction_set,
                          batch_size=self.batch_size,
                          collate_fn=self.collate,
                          num_workers=self.num_workers)
