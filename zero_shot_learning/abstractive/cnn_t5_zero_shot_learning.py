"""
    Resources used:
    https://huggingface.co/docs/transformers/model_doc/t5
    https://www.linkedin.com/pulse/text-summarization-using-hugging-faces-t5-model-arjun-araneta-bhtzc
    https://huggingface.co/docs/transformers/training
    https://tqdm.github.io/
 """

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import time
import torch

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "google-t5/t5-small"

# Load Google T5 pretrained model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load CNN/Daily Mail dataset.
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Load Evaluation Metrics
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

start_time = time.time()

"""
    Data preprocessing: truncate and tokenize inputs.
"""
def preprocess_dataset(dataset, data_subset_size, max_input_length, split):
    print("Starting preprocessing...")
    print(f"Token length: {max_input_length}")

    processed_data = []
    data_subset = dataset[split].select(range(data_subset_size))

    for item in tqdm(data_subset, desc="Preprocessing", total=data_subset_size):
        article = item["article"]
        summary = item["highlights"]

        # Add the 'summarize:' to article input.
        article_with_prompt = "summarize: " + article

        # Tokenize articles & summaries, apply truncation, and ensure that they are padded up until the max length. This
        # yields an object that has a tensor with numerical values.
        inputs = tokenizer(
            article_with_prompt, return_tensors="pt", max_length=max_input_length, truncation=True, padding="max_length"
        )

        processed_data.append({
            "input_ids": inputs["input_ids"].to(device),
            # Since we pad, we want to make sure that during training the focus/attention is non-padded values.
            # "attention_mask": inputs["attention_mask"],
            "labels": summary
        })

    print("\nEnding preprocessing")
    return processed_data


"""
    Compute metrics such as: rouge, meteor, bertscore, and bleu.
"""
def compute_metrics(predictions, references):
    rouge_metrics = rouge.compute(predictions=predictions, references=references)
    meteor_metric = meteor.compute(predictions=predictions, references=references)
    bertscore_metric = bertscore.compute(predictions=predictions, references=references, lang="en")
    bleu_metric = bleu.compute(predictions=predictions, references=references)

    # Needed so rogue 1, 2, etc. are formatted nicely.
    formatted_rouge_metrics = {}
    for key, value in rouge_metrics.items():
        formatted_rouge_metrics[key] = f"{value:.4f}"

    print("\nEvaluation Results:")
    print(f"ROUGE-1: {rouge_metrics['rouge1']}")
    print(f"ROUGE-2: {rouge_metrics['rouge2']}")
    print(f"ROUGE-L: {rouge_metrics['rougeL']}")
    print(f"ROGUE-LSUM: {rouge_metrics['rougeLsum']}")
    print(f"BLEU: {bleu_metric['bleu']:.4f}")
    print(f"METEOR: {meteor_metric['meteor']:.4f}")
    print(f"BERTScore F1: {sum(bertscore_metric['f1']) / len(bertscore_metric['f1']):.4f}")


"""
    Run zero shot learning benchmark.
"""
def zero_shot_learning(preprocessed_data):
    generated_summaries = []
    actual_summaries = []

    for item in tqdm(preprocessed_data, desc="Processing input:", total=len(preprocessed_data)):
        try:
            input_ids = item["input_ids"]

            # Ask model to generate summary.
            summary_ids = model.generate(input_ids=input_ids, max_length=200)

            # Decode predicted summary.
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_summaries.append(generated_summary)

            actual_summary = item["labels"]
            actual_summaries.append(actual_summary)

        except Exception as e:
            print(f"Uh oh error: {e}")
            return

    compute_metrics(generated_summaries, actual_summaries)

processed_test_data = preprocess_dataset(dataset, 2000, 256, "test")

# Running zero shot learning benchmark.
print("\nRunning zero shot learning benchmark...")
zero_shot_learning(processed_test_data)

end_time = time.time()
total_time = end_time - start_time
print(f"Zero shot experiment has completed. Time taken: {total_time:.2f} seconds")