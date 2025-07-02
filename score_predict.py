import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
import evaluate
import random

# Load tokenizer and model
model_checkpoint = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Dummy Data: Replace this with your real dataset
data = [
    {"input": "Score this research question: How does AI affect financial decision-making?", "target": "4"},
    {"input": "Score this research question: Is technology good?", "target": "2"},
    {"input": "Score this research question: What is the effect of virtual labs on programming skill acquisition?", "target": "5"},
    {"input": "Score this research question: Are mobile apps helpful?", "target": "3"},
    {"input": "Score this research question: What is the impact of gamification in online education?", "target": "4"},
]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Tokenize
max_length = 128

def preprocess(example):
    model_input = tokenizer(example["input"], padding="max_length", max_length=max_length, truncation=True)
    labels = tokenizer(example["target"], padding="max_length", max_length=2, truncation=True)

    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_train = dataset["train"].map(preprocess)
tokenized_eval = dataset["test"].map(preprocess)

# Define TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_rq_score_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-4,
    logging_dir="./logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),  # Use FP16 if on GPU
    predict_with_generate=True,
)

# Metric for evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune
trainer.train()

# Save the model
model.save_pretrained("./t5_rq_score_model")
tokenizer.save_pretrained("./t5_rq_score_model")

