"""
"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as1_ol200_ts2000/1e-05/od/checkpoint-1500"
ROUGE-1:     0.3012
ROUGE-2:     0.1131
ROUGE-L:     0.2153
ROUGE-Lsum:  0.2716
BLEU:        0.0895
METEOR:      0.2839
BERTScore F1:0.8672

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as1_ol200_ts2000/1e-05/od/checkpoint-4000"
ROUGE-1:     0.3144
ROUGE-2:     0.1219
ROUGE-L:     0.2248
ROUGE-Lsum:  0.2877
BLEU:        0.0931
METEOR:      0.3102
BERTScore F1:0.8701

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as1_ol200_ts3000/1e-05/od/checkpoint-1500"
ROUGE-1:     0.3010
ROUGE-2:     0.1129
ROUGE-L:     0.2151
ROUGE-Lsum:  0.2697
BLEU:        0.0845
METEOR:      0.2859
BERTScore F1:0.8653

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as1_ol200_ts3000/1e-05/od/checkpoint-5250"
ROUGE-1:     0.3087
ROUGE-2:     0.1195
ROUGE-L:     0.2234
ROUGE-Lsum:  0.2836
BLEU:        0.0957
METEOR:      0.2934
BERTScore F1:0.8700

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as2_ol200_ts2000/1e-05/od/checkpoint-1000"
ROUGE-1:     0.2974
ROUGE-2:     0.1126
ROUGE-L:     0.2096
ROUGE-Lsum:  0.2619
BLEU:        0.0808
METEOR:      0.2941
BERTScore F1:0.8626

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as2_ol200_ts2000/1e-05/od/checkpoint-2250"
ROUGE-1:     0.3025
ROUGE-2:     0.1138
ROUGE-L:     0.2148
ROUGE-Lsum:  0.2752
BLEU:        0.0833
METEOR:      0.2989
BERTScore F1:0.8671

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as4_ol200_ts2000/1e-05/od/checkpoint-750"
ROUGE-1:     0.3003
ROUGE-2:     0.1149
ROUGE-L:     0.2111
ROUGE-Lsum:  0.2616
BLEU:        0.0805
METEOR:      0.2973
BERTScore F1:0.8625

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il512_bs4_as4_ol200_ts2000/1e-05/od/checkpoint-1375"
ROUGE-1:     0.3040
ROUGE-2:     0.1141
ROUGE-L:     0.2135
ROUGE-Lsum:  0.2736
BLEU:        0.0830
METEOR:      0.2980
BERTScore F1:0.8657

##### 256
"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il256_bs4_as1_ol200_ts2000/1e-05/od/checkpoint-1000"
ROUGE-1:     0.2997
ROUGE-2:     0.1154
ROUGE-L:     0.2104
ROUGE-Lsum:  0.2626
BLEU:        0.0772
METEOR:      0.3054
BERTScore F1:0.8634

"E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il256_bs4_as1_ol200_ts2000/1e-05/od/checkpoint-3500"
ROUGE-1:     0.3129
ROUGE-2:     0.1263
ROUGE-L:     0.2258
ROUGE-Lsum:  0.2860
BLEU:        0.1010
METEOR:      0.3017
BERTScore F1:0.8703


"""
from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch
import evaluate
from tqdm import tqdm

best_checkpoint_path = "E:/NLPProject/experiments/cnn_daily_mail/BART/facebook_bart-base_il256_bs4_as1_ol200_ts2000/1e-05/od/checkpoint-3500"
max_input_length = 256
max_output_length = 200
num_test_samples = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
model = BartForConditionalGeneration.from_pretrained(best_checkpoint_path).to(device)
model.eval()

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_data = dataset["test"].select(range(num_test_samples))

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

def generate_summary(text):
    inputs = tokenizer(
        text,
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