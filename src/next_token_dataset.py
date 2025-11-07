import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from configs.config import Config


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, config: Config) -> tuple:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Токенизация текста
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # вход и цель смещаем на 1
        input_seq = input_ids[:-1]
        target_seq = input_ids[1:]

        return input_seq, target_seq, attention_mask[:-1]
