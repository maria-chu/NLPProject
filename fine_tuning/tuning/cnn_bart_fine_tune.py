"""
    Resources:
    1) https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    2) https://insights.encora.com/insights/fine-tuning-large-language-models
    3) https://www.restack.io/p/fine-tuning-answer-t5-huggingface-cat-ai
    4) https://huggingface.co/docs/datasets/en/quickstart
    5) https://huggingface.co/docs/transformers/en/main_classes/callback

    For later:
    - Can use optuna but this will break my GPU lol...super cool library for hyperparameter fine tuning.
    -> https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba
 """
import numpy as np
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import evaluate
import time
import torch

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "facebook/bart-base"

# Load Google T5 tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load CNN/Daily Mail dataset.
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Load Evaluation Metrics
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

learning_rates = [1e-5]
#learning_rates = [4e-5]
start_time = time.time()
max_input_length = 512
training_samples = 3000

"""
    Data preprocessing: truncate and tokenize inputs.
"""
def preprocess_dataset(dataset, data_subset_size, split, max_input_length=max_input_length):
    print("\nStarting preprocessing...")
    print(f"\nToken length: {max_input_length}")

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
        targets = tokenizer(
            summary, return_tensors="pt", max_length=max_input_length, truncation=True, padding="max_length"
        )

        processed_data.append({
            "input_ids": inputs["input_ids"].squeeze(0).to(device),
            # Since we pad, we want to make sure that during training the focus/attention is non-padded values.
            "attention_mask": inputs["attention_mask"].squeeze(0).to(device),
            "labels": targets["input_ids"].squeeze(0).to(device)
        })

    print("\nEnding preprocessing")
    return processed_data


"""
    Compute metrics such as: rouge, meteor, bertscore, and bleu.
"""
def compute_metrics(eval_pred):

    try:
        preds, refs = eval_pred

        predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = tokenizer.batch_decode(refs, skip_special_tokens=True)

        rouge_metrics = rouge.compute(predictions=predictions, references=references)
        meteor_metric = meteor.compute(predictions=predictions, references=references)
        bertscore_metric = bertscore.compute(predictions=predictions, references=references, lang="en")
        bleu_metric = bleu.compute(predictions=predictions, references=references)

        # They have to be of type float otherwise, tensorboard will complain...
        return {
            "ROUGE-1": rouge_metrics['rouge1'],
            "ROGUE-2": rouge_metrics['rouge2'],
            "ROGUE-L": rouge_metrics['rougeL'],
            "ROGUE-LSUM": rouge_metrics['rougeLsum'],
            "BLUE": bleu_metric["bleu"],
            "METEOR": meteor_metric["meteor"],
            "BERTScore F1": sum(bertscore_metric["f1"]) / len(bertscore_metric["f1"])
        }
    except Exception as e:
        print(f"Uh oh exception occurred for metrics: {e}")

        # Return 0 for metrics - maybe this issue occurs because of training...
        return {
            "ROUGE-1": 0,
            "ROGUE-2": 0,
            "ROGUE-L": 0,
            "ROGUE-LSUM": 0,
            "BLUE": 0,
            "METEOR": 0,
            "BERTScore F1": 0
        }


"""
    Set hyperparameters and train model with different learning rates. Evaluate model using validation set.
"""
def train_model(selected_learning_rate):
    # Need to do this so that the model is reinitialized after every learning rate.
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    model.config.use_cache = False  # Needed to get of checkpoint warnings.
    model.config.use_reentrant = False  # Needed for reetrant warnings.

    # Makes sure data batches are consistent.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    batch_size = 4
    accumulation_steps = 1

    # This allows for the generated summaries to be fixed length and improves metrics! Yay!
    # Picked 200 because between training and validation samples the max was below 200.
    generation_max_length = 200
    current_model = model_name.replace('/', '_')
    main_dir = rf"E:/NLPProject/experiments/cnn/{current_model}_il{str(max_input_length)}_bs{str(batch_size)}_as{accumulation_steps}_ol{str(generation_max_length)}_ts{training_samples}/{str(selected_learning_rate)}"
    output_dir = rf"{main_dir}/od"
    logging_dir = rf"{main_dir}/ld"

    # Define the models hyperparameters.
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=selected_learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        predict_with_generate=True,
        logging_dir=logging_dir,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        # Get rid of deprecation warning for AdamW optimizer. Come back to this...seeing issues.
        #optim="adamw_torch"
        generation_max_length=generation_max_length,
        # Optimization options below cuz I can't update batch size past 4...
        #optim='adamw_bnb_8bit' # Fails...this is for a Linux based system - come back to this
        gradient_accumulation_steps=accumulation_steps, # Cool - if this is set to 2 then it would simulate a batch size of 8
        #bf16 = True, # Not supported for my GPU version...
        gradient_checkpointing = True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Prevents stopping too early and underfitting.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train model.
    trainer.train()

    # Evaluate model.
    val_results = trainer.evaluate(eval_dataset=val_dataset)

    print("\nMetrics for Validation:")
    print(val_results)


# Preprocessed dataset splits and conversion from list type to dataset object.
processed_train_data = preprocess_dataset(dataset, training_samples, "train")
processed_val_data = preprocess_dataset(dataset, 500, "validation")

train_dataset = Dataset.from_list(processed_train_data)
val_dataset = Dataset.from_list(processed_val_data)

# Train
for learning_rate in tqdm(learning_rates, desc="Learning rates", total=len(learning_rates)):
    train_model(learning_rate)

end_time = time.time()
total_time = end_time - start_time
print(f"Best learning rate experiment has completed. Time taken: {total_time:.2f} seconds")