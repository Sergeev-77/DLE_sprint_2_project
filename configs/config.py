import torch


class Config:
    def __init__(self, is_test=False):
        self.raw_ds_path = "./data/raw_dataset.txt"
        self.ds_processed_path = "./data/dataset_processed.txt"
        self.ds_train_path = "./data/train.csv"
        self.ds_val_path = "./data/val.csv"
        self.ds_test_path = "./data/test.csv"
        self.model_path = "./models/lstm_model.pth"
        self.max_raw__ds_length = (
            100 if is_test else None
        )  # на отладке кода ограничим длину исходного датасета, на бою none
        self.batch_size = (
            16 if is_test else 256
        )  # на отладке кода ограничим батч
        self.embedding_dim = 256
        self.hidden_dim = 128
        self.num_layers = 1
        self.dropout = 0.3
        self.learning_rate = 0.002
        self.num_epochs = 10
        self.max_length = 64
        self.train_val_split = 0.8
        self.text_split_on_prediction = 0.75
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_name = "distilgpt2"
        self.pad_token_id = 0
        self.pad_token = ""
        self.vocab_size = 0
