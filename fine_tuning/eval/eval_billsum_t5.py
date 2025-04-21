"""
5 beams
https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News

E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-9500
ROUGE-1:     0.4048
ROUGE-2:     0.2383
ROUGE-L:     0.3143
ROUGE-Lsum:  0.3395
BLEU:        0.1071
METEOR:      0.2716
BERTScore F1:0.8706

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-10000"
ROUGE-1:     0.4044
ROUGE-2:     0.2385
ROUGE-L:     0.3141
ROUGE-Lsum:  0.3394
BLEU:        0.1083
METEOR:      0.2718
BERTScore F1:0.8704

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts3000/5e-05/od/checkpoint-14250"
ROUGE-1:     0.4054
ROUGE-2:     0.2436
ROUGE-L:     0.3167
ROUGE-Lsum:  0.3420
BLEU:        0.1069
METEOR:      0.2733
BERTScore F1:0.8726

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as1_ol200_ts3000/5e-05/od/checkpoint-15000"
ROUGE-1:     0.4078
ROUGE-2:     0.2457
ROUGE-L:     0.3186
ROUGE-Lsum:  0.3443
BLEU:        0.1075
METEOR:      0.2737
BERTScore F1:0.8733

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as2_ol200_ts2000/5e-05/od/checkpoint-4750"
ROUGE-1:     0.4007
ROUGE-2:     0.2314
ROUGE-L:     0.3087
ROUGE-Lsum:  0.3357
BLEU:        0.1058
METEOR:      0.2688
BERTScore F1:0.8681

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as2_ol200_ts2000/5e-05/od/checkpoint-5000"
ROUGE-1:     0.4004
ROUGE-2:     0.2315
ROUGE-L:     0.3096
ROUGE-Lsum:  0.3357
BLEU:        0.1075
METEOR:      0.2694
BERTScore F1:0.8681

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as4_ol200_ts2000/5e-05/od/checkpoint-2375"
ROUGE-1:     0.3992
ROUGE-2:     0.2273
ROUGE-L:     0.3076
ROUGE-Lsum:  0.3332
BLEU:        0.1076
METEOR:      0.2686
BERTScore F1:0.8663

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il512_bs4_as4_ol200_ts2000/5e-05/od/checkpoint-2500"
ROUGE-1:     0.3983
ROUGE-2:     0.2273
ROUGE-L:     0.3069
ROUGE-Lsum:  0.3334
BLEU:        0.1073
METEOR:      0.2689
BERTScore F1:0.8663

##### 256
"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-9500"
ROUGE-1:     0.3734
ROUGE-2:     0.2128
ROUGE-L:     0.2937
ROUGE-Lsum:  0.3172
BLEU:        0.0784
METEOR:      0.2450
BERTScore F1:0.8667

"E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-10000"
ROUGE-1:     0.3725
ROUGE-2:     0.2133
ROUGE-L:     0.2939
ROUGE-Lsum:  0.3169
BLEU:        0.0762
METEOR:      0.2436
BERTScore F1:0.8672


"""
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import evaluate
from tqdm import tqdm

# === CONFIG ===
best_checkpoint_path = "E:/NLPProject/experiments/bill_sum/t5/google-t5_t5-small_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-10000"
max_input_length = 256
max_output_length = 200
num_test_samples = 500

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(best_checkpoint_path).to(device)
model.eval()

dataset = load_dataset("billsum")
test_data = dataset["test"].select(range(500))

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

print(f"\nGenerating summaries for {num_test_samples} test samples...\n")

predictions = []
references = []

for item in tqdm(test_data, total=num_test_samples):
    generated_summary = generate_summary(item["text"])
    predictions.append(generated_summary)
    references.append(item["summary"])

rouge_metrics = rouge.compute(predictions=predictions, references=references)
meteor_metric = meteor.compute(predictions=predictions, references=references)
bertscore_metric = bertscore.compute(predictions=predictions, references=references, lang="en")
bleu_metric = bleu.compute(predictions=predictions, references=references)

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
