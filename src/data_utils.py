from configs.config import Config
import re
import csv
import emoji
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def process_dataset(
    config: Config,
) -> list[str]:
    """очистка текстов"""
    input_file_path: str = config.raw_ds_path
    output_file_path: str = config.ds_processed_path
    max_raw_lenght: int | None = config.max_raw__ds_length

    def _clean_text(text):
        text = re.sub(r"@\w+", " ", text)  # убрать упоминания
        text = emoji.replace_emoji(text, replace="")  # убрать эмодзи
        text = re.sub(r"http\S+", " ", text)  # убрать ссылки
        # оставить только буквы и цифры
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
        text = text.lower()  # к нижнему регистру
        return text

    with open(input_file_path, "r") as f:
        corpus = [line for line in f.read().strip().split("\n")][
            :max_raw_lenght
        ]

    corpus_clean = [
        line
        for line in [_clean_text(line) for line in corpus]
        if line and len(line.split()) > 1  # не менее двух слов в тексте
    ]

    del corpus

    word_counts = [len(text.split()) for text in corpus_clean]

    print(f"\nколичество текстов: {len(corpus_clean)}")
    print("статистика количества слов в тексте:")
    print(f"мин: {np.min(word_counts):.0f}")
    print(f"медиана: {np.median(word_counts):.2f}")
    print(f"среднее: {np.mean(word_counts):.2f}")
    print(f"макс: {np.max(word_counts):.0f}")
    print(f"5-й перцентиль: {np.percentile(word_counts, 5):.2f}")
    print(f"95-й перцентиль: {np.percentile(word_counts, 95):.2f}")

    plt.hist(word_counts, bins=50, edgecolor="black")
    plt.title("Распределение количества слов в текстах")
    plt.xlabel("количество слов")
    plt.ylabel("количество текстов")
    plt.grid(True)
    plt.show()
    print("\nпримеры чистых текстов:\n")
    for i in range(5):
        print(corpus_clean[i])
    if not output_file_path:
        return corpus_clean

    with open(output_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for line in corpus_clean:
            if len(line):
                writer.writerow([line])
    return corpus_clean


def split_dataset(texts: list[str], config: Config) -> tuple:
    """разбиение текста на подвыборки"""

    train_texts, temp_texts = train_test_split(
        texts, train_size=config.train_val_split, random_state=42
    )
    val_texts, test_texts = train_test_split(
        temp_texts, test_size=0.5, random_state=42
    )

    train_df = pd.DataFrame({"text": train_texts})
    val_df = pd.DataFrame({"text": val_texts})
    test_df = pd.DataFrame({"text": test_texts})

    train_df.to_csv(config.ds_train_path, index=False)
    val_df.to_csv(config.ds_val_path, index=False)
    test_df.to_csv(config.ds_test_path, index=False)

    return train_texts, val_texts, test_texts
