import torch
import torch.nn as nn
from configs.config import Config
from rouge_score import rouge_scorer


class LSTMModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        embedding_dim = config.embedding_dim
        vocab_size = config.vocab_size
        dropout = config.dropout

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=config.pad_token_id
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if self.num_layers > 1 else 0,
        )
        self.fc = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)

        if hidden is None:
            lstm_out, hidden = self.lstm(emb)
        else:
            lstm_out, hidden = self.lstm(emb, hidden)

        output = self.fc(self.dropout(lstm_out))
        return output, hidden


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def generate_text(
    model, tokenizer, input_tokens, config: Config, max_length=None
):
    model.eval()

    if max_length is None:
        max_length = config.max_length

    generated = (
        input_tokens.tolist()
        if isinstance(input_tokens, torch.Tensor)
        else input_tokens.copy()
    )

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([generated]).to(config.device)
            output, _ = model(input_tensor)

            # Берем логиты для последнего токена
            next_token_logits = output[0, -1, :]

            # Применяем softmax и выбираем следующий токен
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()

            generated.append(next_token)

            # Останавливаемся если достигли конца текста
            if next_token == config.pad_token_id:
                break

    # Декодируем сгенерированную часть
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    input_text = tokenizer.decode(
        (
            input_tokens.tolist()
            if isinstance(input_tokens, torch.Tensor)
            else input_tokens
        ),
        skip_special_tokens=True,
    )

    # только сгенерированная часть
    return generated_text[len(input_text) :], input_text


def generate_samples(model, tokenizer, config, samples: list[str]):
    """генерация автодополнения для списка текстов
    по text_split_on_prediction начальных слов текста"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    for text in samples:
        words = text.split()
        slice = int(len(words) * config.text_split_on_prediction)

        input_txt = " ".join(words[:slice])
        target_txt = " ".join(words[slice:])

        input_ids = tokenizer(
            input_txt,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            # return_tensors="pt",
        )["input_ids"]

        print("asis:      ", input_txt, " -> ", target_txt)
        gen_txt, input_txt = generate_text(
            model, tokenizer, input_ids, config=config
        )
        print("generated: ", input_txt, " -> ", gen_txt)
        scores = scorer.score(target_txt, gen_txt)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        print(f"rouge1: {r1:.4f},   rouge2: {r2:.4f}")
        print()
