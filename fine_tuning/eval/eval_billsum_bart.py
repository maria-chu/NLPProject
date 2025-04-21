"""
5 beams
https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News

"E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-7000"
ROUGE-1:     0.4135
ROUGE-2:     0.2368
ROUGE-L:     0.3131
ROUGE-Lsum:  0.3475
BLEU:        0.0973
METEOR:      0.2763
BERTScore F1:0.8773

"E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-4500"
ROUGE-1:     0.4160
ROUGE-2:     0.2337
ROUGE-L:     0.3107
ROUGE-Lsum:  0.3468
BLEU:        0.1031
METEOR:      0.2764
BERTScore F1:0.8774

"E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as1_ol200_ts3000/5e-05/od/checkpoint-9750"
ROUGE-1:     0.4071
ROUGE-2:     0.2334
ROUGE-L:     0.3114
ROUGE-Lsum:  0.3419
BLEU:        0.0946
METEOR:      0.2696
BERTScore F1:0.8769

E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as1_ol200_ts3000/5e-05/od/checkpoint-6000
ROUGE-1:     0.4059
ROUGE-2:     0.2390
ROUGE-L:     0.3140
ROUGE-Lsum:  0.3432
BLEU:        0.0873
METEOR:      0.2639
BERTScore F1:0.8781

E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as2_ol200_ts2000/5e-05/od/checkpoint-4250
ROUGE-1:     0.4065
ROUGE-2:     0.2333
ROUGE-L:     0.3062
ROUGE-Lsum:  0.3376
BLEU:        0.0937
METEOR:      0.2697
BERTScore F1:0.8764

E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as2_ol200_ts2000/5e-05/od/checkpoint-3000
ROUGE-1:     0.4252
ROUGE-2:     0.2353
ROUGE-L:     0.3133
ROUGE-Lsum:  0.3516
BLEU:        0.1157
METEOR:      0.2892
BERTScore F1:0.8765

"E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as4_ol200_ts2000/5e-05/od/checkpoint-2125"
ROUGE-1:     0.4131
ROUGE-2:     0.2377
ROUGE-L:     0.3131
ROUGE-Lsum:  0.3447
BLEU:        0.0982
METEOR:      0.2743
BERTScore F1:0.8778

E:/NLPProject/experiments/bill_sum/facebook_bart-base_il512_bs4_as4_ol200_ts2000/5e-05/od/checkpoint-1500
ROUGE-1:     0.4110
ROUGE-2:     0.2361
ROUGE-L:     0.3122
ROUGE-Lsum:  0.3453
BLEU:        0.0899
METEOR:      0.2725
BERTScore F1:0.8783

##### 256
E:/NLPProject/experiments/bill_sum/facebook_bart-base_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-3500
ROUGE-1:     0.3934
ROUGE-2:     0.2104
ROUGE-L:     0.2940
ROUGE-Lsum:  0.3232
BLEU:        0.0899
METEOR:      0.2581
BERTScore F1:0.8722

E:/NLPProject/experiments/bill_sum/facebook_bart-base_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-6000
ROUGE-1:     0.3946
ROUGE-2:     0.2064
ROUGE-L:     0.2920
ROUGE-Lsum:  0.3258
BLEU:        0.0936
METEOR:      0.2603
BERTScore F1:0.8719
"""
from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch
import evaluate
from tqdm import tqdm


def print_test_results(best_checkpoint_path):
    max_input_length = 256
    max_output_length = 200
    num_test_samples = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
    model = BartForConditionalGeneration.from_pretrained(best_checkpoint_path).to(device)
    model.eval()

    dataset = load_dataset("billsum")
    test_data = dataset["test"].select(range(500))

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

checkpoints = [
    "E:/NLPProject/experiments/bill_sum/facebook_bart-base_il256_bs4_as1_ol200_ts2000/5e-05/od/checkpoint-6000"
]

for checkpoint in checkpoints:
    print_test_results(checkpoint)