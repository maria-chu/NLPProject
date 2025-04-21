"""
Resources:
https://pypi.org/project/bert-extractive-summarizer/
"""
from summarizer import Summarizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test").select(range(500))

bert_model = Summarizer()

generated_summaries = []
reference_summaries = []

for article in tqdm(dataset, desc="Summarizing"):
    text = article['article']
    reference_summary = article['highlights']
    summary = ''.join(bert_model(text, num_sentences=3, max_length=256))
    generated_summaries.append(summary)
    reference_summaries.append(reference_summary)

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

rouge_score = rouge.compute(predictions=generated_summaries, references=reference_summaries)
meteor_score = meteor.compute(predictions=generated_summaries, references=reference_summaries)
bertscore_score = bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en")
bleu_score = bleu.compute(predictions=generated_summaries, references=reference_summaries)

formatted_rouge_metrics = {}
for key, value in rouge_score.items():
    formatted_rouge_metrics[key] = f"{value:.4f}"

print("\nEvaluation Results:")
print(f"ROUGE-1: {formatted_rouge_metrics['rouge1']}")
print(f"ROUGE-2: {formatted_rouge_metrics['rouge2']}")
print(f"ROUGE-L: {formatted_rouge_metrics['rougeL']}")
print(f"ROGUE-LSUM: {formatted_rouge_metrics['rougeLsum']}")
print(f"BLEU: {bleu_score['bleu']:.4f}")
print(f"METEOR: {meteor_score['meteor']:.4f}")
print(f"BERTScore F1: {sum(bertscore_score['f1']) / len(bertscore_score['f1']):.4f}")