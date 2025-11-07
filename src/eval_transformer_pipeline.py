from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from datasets import Dataset


def calculate_rouge_transformer(
    generator, config, samples: list[str], prefix="test"
):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    dataset = Dataset.from_dict({"text": samples})

    progress_bar = tqdm(dataset, desc=f"         сalc rouge metrics {prefix}")

    for item in progress_bar:
        text = item["text"]
        words = text.split()
        slice = int(len(words) * config.text_split_on_prediction)

        input_txt = " ".join(words[:slice])
        target_txt = " ".join(words[slice:])
        result = generator(
            input_txt,
            max_new_tokens=20,
            do_sample=True,
            truncation=True,
            pad_token_id=config.pad_token_id,
        )
        gen_txt = result[0]["generated_text"][len(input_txt) :].strip()

        # print("generated: ", input_txt, " -> ", gen_txt)

        scores = scorer.score(target_txt, gen_txt)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        progress_bar.set_postfix({"rouge1 max": f"{np.max(rouge1_scores):.4f}"})
    return np.mean(rouge1_scores), np.mean(rouge2_scores)
