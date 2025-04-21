"""
"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts3000/4e-05/od/checkpoint-3000"
ROUGE-1:     0.3052
ROUGE-2:     0.1195
ROUGE-L:     0.2263
ROUGE-Lsum:  0.2597
BLEU:        0.0956
METEOR:      0.2952
BERTScore F1:0.8665

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts3000/4e-05/od/checkpoint-6750"
ROUGE-1:     0.3058
ROUGE-2:     0.1188
ROUGE-L:     0.2235
ROUGE-Lsum:  0.2588
BLEU:        0.0930
METEOR:      0.2982
BERTScore F1:0.8662

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as2_ol200/4e-05/od/checkpoint-1000"
ROUGE-1:     0.3046
ROUGE-2:     0.1198
ROUGE-L:     0.2233
ROUGE-Lsum:  0.2573
BLEU:        0.0947
METEOR:      0.2938
BERTScore F1:0.8649

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as2_ol200/4e-05/od/checkpoint-2250"
ROUGE-1:     0.3052
ROUGE-2:     0.1193
ROUGE-L:     0.2247
ROUGE-Lsum:  0.2587
BLEU:        0.0953
METEOR:      0.2976
BERTScore F1:0.8666

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as4_ol200/4e-05/od/checkpoint-750"
ROUGE-1:     0.3003
ROUGE-2:     0.1171
ROUGE-L:     0.2230
ROUGE-Lsum:  0.2554
BLEU:        0.0920
METEOR:      0.2917
BERTScore F1:0.8637

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_as4_ol200/4e-05/od/checkpoint-1375"
ROUGE-1:     0.3024
ROUGE-2:     0.1167
ROUGE-L:     0.2218
ROUGE-Lsum:  0.2551
BLEU:        0.0928
METEOR:      0.2919
BERTScore F1:0.8660

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_ol200/4e-05/od/checkpoint-1500"
ROUGE-1:     0.3041
ROUGE-2:     0.1198
ROUGE-L:     0.2243
ROUGE-Lsum:  0.2572
BLEU:        0.0952
METEOR:      0.2943
BERTScore F1:0.8661

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il512_bs4_ol200/4e-05/od/checkpoint-4000"
ROUGE-1:     0.2986
ROUGE-2:     0.1150
ROUGE-L:     0.2184
ROUGE-Lsum:  0.2516
BLEU:        0.0921
METEOR:      0.2868
BERTScore F1:0.8654

###### 256
"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il256_bs4_ol200/4e-05/od/checkpoint-2000"
ROUGE-1:     0.2994
ROUGE-2:     0.1193
ROUGE-L:     0.2211
ROUGE-Lsum:  0.2538
BLEU:        0.0970
METEOR:      0.2906
BERTScore F1:0.8655

"E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il256_bs4_ol200/4e-05/od/checkpoint-4500"
ROUGE-1:     0.3003
ROUGE-2:     0.1191
ROUGE-L:     0.2217
ROUGE-Lsum:  0.2544
BLEU:        0.0973
METEOR:      0.2937
BERTScore F1:0.8655

"""
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import evaluate
from tqdm import tqdm

best_checkpoint_path = "E:/NLPProject/experiments/cnn_daily_mail/t5/google-t5_t5-small_il256_bs4_ol200/4e-05/od/checkpoint-4500"
max_input_length = 256
max_output_length = 200
num_test_samples = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(best_checkpoint_path).to(device)
model.eval()

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_data = dataset["test"].select(range(num_test_samples))

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

def generate_summary(text):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    ).to(device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output_length,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\nGenerating summaries for {num_test_samples} CNN/DailyMail test samples...\n")

predictions = []
references = []

for item in tqdm(test_data, total=num_test_samples):
    generated_summary = generate_summary(item["article"])
    predictions.append(generated_summary)
    references.append(item["highlights"])

rouge_metrics = rouge.compute(predictions=predictions, references=references)
meteor_metric = meteor.compute(predictions=predictions, references=references)
bertscore_metric = bertscore.compute(predictions=predictions, references=references, lang="en")
bleu_metric = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

print("Evaluation Results:")
# Needed so rogue 1, 2, etc. are formatted nicely.
formatted_rouge_metrics = {}
for key, value in rouge_metrics.items():
    formatted_rouge_metrics[key] = f"{value:.4f}"

print(f"ROUGE-1: {rouge_metrics['rouge1']}")
print(f"ROUGE-2: {rouge_metrics['rouge2']}")
print(f"ROUGE-L: {rouge_metrics['rougeL']}")
print(f"ROGUE-LSUM: {rouge_metrics['rougeLsum']}")
print(f"BLEU: {bleu_metric['bleu']:.4f}")
print(f"METEOR: {meteor_metric['meteor']:.4f}")
print(f"BERTScore F1: {sum(bertscore_metric['f1']) / len(bertscore_metric['f1']):.4f}")
