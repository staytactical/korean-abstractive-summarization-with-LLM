import pandas as pd
from transformers import AutoTokenizer
import datasets
from torch.utils.data import DataLoader
from transformers import default_data_collator

def train_batch_preprocess(tokenizer, batch):
    inputs = batch['text']
    targets = batch['summary']
    model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def prepare_train_set(data_path, tokenizer):
    train_df = pd.read_csv(data_path)
    train_set = datasets.Dataset.from_pandas(train_df)
    train_set = train_set.map(
        lambda batch: train_batch_preprocess(tokenizer, batch),
        remove_columns = ['id', 'text', 'summary'],
        batched = True,
        batch_size = 1000,
    )
    return train_set

def create_train_loader(train_set, batch_size):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=default_data_collator,
    )
    return train_loader
