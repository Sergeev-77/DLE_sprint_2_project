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
    inputs = []
    targets = []
   
    for text in samples:
        words = text.split()
        slice = int(len(words) * config.text_split_on_prediction)

        input_txt = " ".join(words[:slice])
        target_txt = " ".join(words[slice:])
        inputs.append(input_txt)
        targets.append(target_txt)

    results = generator(
            inputs,
            max_new_tokens=config.max_length,
            do_sample=True,
            pad_token_id=config.pad_token_id,
        )
    for i, result in enumerate(results):
        full_text = result[0]["generated_text"]
        input_txt = inputs[i]
        target_txt = targets[i]

        gen_txt = full_text[len(input_txt) :].strip()

        scores = scorer.score(target_txt, gen_txt)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        # progress_bar.set_postfix({"rouge1 max": f"{np.max(rouge1_scores):.4f}"})
    return np.mean(rouge1_scores), np.mean(rouge2_scores)
