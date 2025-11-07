import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from configs.config import Config
from src.lstm_model import generate_text


def calculate_rouge_lstm(
    model, dataloader, tokenizer, config: Config, prefix="train"
):
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader, desc=f"         сalc rouge metrics {prefix}"
        )
        for input_seq, target_seq, attention_mask in progress_bar:

            input_seq = input_seq.to(config.device)
            target_seq = target_seq.to(config.device)
            attention_mask = attention_mask.to(config.device)

            batch_size = input_seq.size(0)

            for i in range(batch_size):
                mask_i = attention_mask[i]
                real_length = mask_i.sum().item()  # длина исходного текста
                split_idx = int(
                    real_length * config.text_split_on_prediction
                )  # часть исходного текста
                if split_idx < 1:
                    continue  # нужно, чтобы был хотя бы один токен в input

                # берем первую часть как вход для генерации продолжения
                input_ids = input_seq[i, :split_idx]
                # таргет для оценки (в датасете смещен на 1-у позицию)
                target_ids = target_seq[i, split_idx - 1 :]

                # генерация продолжения
                gen_text, _ = generate_text(
                    model,
                    tokenizer,
                    input_ids,
                    max_length=config.max_length - split_idx,
                    config=config,
                )

                # оригинальное продолжение
                org_text = tokenizer.decode(
                    target_ids, skip_special_tokens=True
                )

                # скоры
                scores = scorer.score(org_text, gen_text)
                r1 = scores["rouge1"].fmeasure
                r2 = scores["rouge2"].fmeasure
                rouge1_scores.append(r1)
                rouge2_scores.append(r2)
            progress_bar.set_postfix(
                {"rouge1 max": f"{np.max(rouge1_scores):.4f}"}
            )
    return np.mean(rouge1_scores), np.mean(rouge2_scores)
