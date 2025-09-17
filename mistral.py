import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import json
import os
import re
from charts import plot_training_history_matplotlib, generate_training_history_chartjs
from huggingface_hub import login

# Authenticate with Hugging Face
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set. Please set it in the SLURM script.")
login(token=hf_token)
print("Hugging Face login successful.")

class MistralRQsDataset(Dataset):
    """Dataset for Mistral fine-tuning for scoring and RQ generation"""
    
    def __init__(self, prompts, tokenizer, max_length=512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]['prompt']
        target = self.prompts[idx]['target']
        
        # Combine prompt and target for causal language modeling
        full_text = prompt + target
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal LM, input_ids and labels are the same
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Create labels (same as input_ids, but with -100 for padding tokens)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_few_shot_prompts(df):
    """Create few-shot prompts for scoring and RQ generation"""
    
    scoring_examples = []
    generation_examples = []
    
    for _, row in df.sample(3, random_state=42).iterrows():
        prompt = f"""Score the following research question on a scale of 1–5 for each criterion below 
(1 = very poor, 5 = excellent). Provide integer values only.

Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.


Question: {row['RQ']}
Title: {row['Title']}

Output only the JSON scores, e.g., {{"Relevance": 4, "Fluency": 3, "Feasibility": 5, "Free of vagueness": 4}}."""
        target = f"""{{ "Relevance": {row['Relevance']}, "Fluency": {row['Fluency']}, "Feasibility": {row['Feasibility']}, "Free of vagueness": {row['Free of vagueness']} }}"""
        scoring_examples.append({'prompt': prompt, 'target': target})
    
    #learning the mapping training
    df_bad = df[(df['Bad RQ'] == 1) & df['Improved RQ Suggestion'].notna()]
    for _, row in df_bad.sample(min(3, len(df_bad)), random_state=42).iterrows():
        comment = row['Comments'] if pd.notna(row['Comments']) else ""
        prompt = f"""Improve the following research question. Use the title and feedback (if available) as context. Output only the improved research question, ending with a question mark. 
        The improved RQ should be meeting all of the following criteria: 
        Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.

Output only ONE improved research question.
- It must be concise, fluent, and end with a "?"


Original Question: {row['RQ']}
Title: {row['Title']}
Feedback: {comment}"""
        target = row['Improved RQ Suggestion']
        generation_examples.append({'prompt': prompt, 'target': target})
    
    return scoring_examples, generation_examples

def load_and_preprocess_data(excel_file_path):
    """Load and preprocess the CleanedRQTrain.xlsx file"""
    
    print(f"Loading data from {excel_file_path}...")
    try:
        df = pd.read_excel(excel_file_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    print(f"Loaded {len(df)} research questions")
    print("Raw columns:", df.columns.tolist())
    
    df.columns = [col.strip().replace('\xa0', ' ') for col in df.columns]
    print("Normalized columns:", df.columns.tolist())
    
    required_columns = ['RQ', 'Title', 'Relevance', 'Fluency', 'Feasibility', 'Free of vagueness', 
                       'Bad RQ', 'Good RQ', 'Perfect RQ', 'Improved RQ Suggestion', 'Comments']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return None
    
    print("\nRaw data summary:")
    print(f"Rows with Bad RQ == 'y': {len(df[df['Bad RQ'] == 'y'])}")
    print(f"Rows with Bad RQ == 'n': {len(df[df['Bad RQ'] == 'n'])}")
    print(f"Rows with Good RQ == 'y': {len(df[df['Good RQ'] == 'y'])}")
    print(f"Rows with Perfect RQ == 'y': {len(df[df['Perfect RQ'] == 'y'])}")
    print(f"Rows with non-null Improved RQ Suggestion: {len(df[df['Improved RQ Suggestion'].notna()])}")
    
    for col in ['Bad RQ', 'Good RQ', 'Perfect RQ']:
        print(f"\nUnique values in {col} before conversion:", df[col].unique().tolist())
        df[col] = df[col].map({'y': 1, 'n': 0}).fillna(0).astype(int)
        print(f"Unique values in {col} after conversion:", df[col].unique().tolist())
    
    initial_len = len(df)
    df = df.dropna(subset=['RQ', 'Title'])
    print(f"\nAfter dropping missing RQ/Title: {len(df)} rows (dropped {initial_len - len(df)})")
    
    score_columns = ['Relevance', 'Fluency', 'Feasibility', 'Free of vagueness']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').clip(1, 5)
    
    initial_len = len(df)
    df = df.dropna(subset=score_columns)
    print(f"After dropping missing scores: {len(df)} rows (dropped {initial_len - len(df)})")
    
    improved_count = df['Improved RQ Suggestion'].notna().sum()
    print(f"Questions with improvements: {improved_count}")
    
    print("\nFiltered data summary:")
    print(f"Rows with Bad RQ == 1: {len(df[df['Bad RQ'] == 1])}")
    print(f"Rows with Bad RQ == 0: {len(df[df['Bad RQ'] == 0])}")
    print(f"Rows with Good RQ == 1: {len(df[df['Good RQ'] == 1])}")
    print(f"Rows with Perfect RQ == 1: {len(df[df['Perfect RQ'] == 1])}")
    print(f"Rows with non-null Improved RQ Suggestion: {len(df[df['Improved RQ Suggestion'].notna()])}")
    
    return df

def train_mistral_model(df, model_save_path='models/mistral_model'):
    """Fine-tune Mistral-7B for scoring and RQ generation"""
    
    print("=== Training Mistral-7B Model ===")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize tokenizer and model
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #hf_token = "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"  # my Hugging Face token
    
    print(f"Loading model: {model_name}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token to eos_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hf_token, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Resize token embeddings if pad_token was added
    if tokenizer.pad_token == tokenizer.eos_token:
        # No need to resize embeddings since we're using existing eos_token
        pass
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Create few-shot prompts
    scoring_examples, generation_examples = create_few_shot_prompts(df)
    
    # Prepare training data
    prompts = []
    
    # Scoring prompts (all rows)
    for _, row in df.iterrows():
        prompt = f"""[INST] Below are example research question scorings:
{chr(10).join([ex['prompt'] + chr(10) + 'Output: ' + ex['target'] for ex in scoring_examples])}

Score the following research question on a scale of 1–5 for each criterion below 
(1 = very poor, 5 = excellent). Provide integer values only.

Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.

Question: {row['RQ']}
Title: {row['Title']}

Output only the JSON scores, e.g., {{"Relevance": 4, "Fluency": 3, "Feasibility": 5, "Free of vagueness": 4}}. [/INST]"""

        target = f"""{{ "Relevance": {row['Relevance']}, "Fluency": {row['Fluency']}, "Feasibility": {row['Feasibility']}, "Free of vagueness": {row['Free of vagueness']} }}"""
        prompts.append({'prompt': prompt, 'target': target})
    
    # RQ generation prompts
    df_bad = df[(df['Bad RQ'] == 1) & df['Improved RQ Suggestion'].notna()].copy()
    print(f"Bad RQs with improvements: {len(df_bad)}")
    
    df_good = df[(df['Bad RQ'] == 0) & ((df['Good RQ'] == 1) | (df['Perfect RQ'] == 1))].copy()
    print(f"Good/Perfect RQs before self-supervision: {len(df_good)}")
    df_good.loc[df_good['Improved RQ Suggestion'].isna(), 'Improved RQ Suggestion'] = df_good['RQ']
    print(f"Good/Perfect RQs (self-supervised): {len(df_good)}")
    
    df_improved = pd.concat([df_bad, df_good], ignore_index=True)
    print(f"Total improvement examples: {len(df_improved)}")
    
    if len(df_improved) == 0:
        print("ERROR: No data for RQ generation. Skipping generation training.")
    else:
        for _, row in df_improved.iterrows():
            comment = row['Comments'] if pd.notna(row['Comments']) else ""
            prompt = f"""[INST] Below are example research question improvements:
{chr(10).join([ex['prompt'] + chr(10) + 'Output: ' + ex['target'] for ex in generation_examples])}

Improve the following research question. Use the title and feedback (if available) as context. Output only the improved research question, ending with a question mark.Improve the following research question. Use the title and feedback (if available) as context. Output only the improved research question, ending with a question mark. 
        The improved RQ should be meeting all of the following criteria: 
        Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.

Output only ONE improved research question.
- It must be concise, fluent, and end with a "?"

Original Question: {row['RQ']}
Title: {row['Title']}
Feedback: {comment} [/INST]"""

            target = row['Improved RQ Suggestion']
            prompts.append({'prompt': prompt, 'target': target})
    
    # Split data
    train_prompts, test_prompts = train_test_split(prompts, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_prompts)}")
    print(f"Test samples: {len(test_prompts)}")
    
    # Create datasets
    train_dataset = MistralRQsDataset(train_prompts, tokenizer)
    test_dataset = MistralRQsDataset(test_prompts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size to avoid memory issues
        per_device_eval_batch_size=1,   # Reduced batch size to avoid memory issues
        gradient_accumulation_steps=4,  # Compensate for smaller batch size
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f'{model_save_path}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        dataloader_pin_memory=False,  # Disable pin_memory to avoid the warning
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (newer API)
    )
    
    training_result = trainer.train()
    
    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Extract losses
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    training_history = {
        'model_type': 'mistral_7b',
        'train_losses': train_losses,
        'test_losses': eval_losses,
        'train_size': len(train_prompts),
        'test_size': len(test_prompts)
    }
    
    with open(f"{model_save_path}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Generate plots using charts.py
    plot_training_history_matplotlib(training_history, 'Mistral-7B Model', f"{model_save_path}/training_history.png")
    generate_training_history_chartjs(training_history, 'Mistral-7B Model', f"{model_save_path}/training_history_chart.json")
    
    print(f"Mistral-7B model saved to: {model_save_path}")
    return model, tokenizer, training_history

def test_mistral_model(df, model_save_path='models/mistral_model', output_excel='Mistral_Test_Results.xlsx'):
    """Test Mistral-7B model on CleanedRQTrain.xlsx and save results to Excel"""
    
    print("=== Testing Mistral-7B Model ===")
    
    # Load model and tokenizer
    try:
        print("Loading saved model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_save_path, token =hf_token)
        # Ensure pad_token is set when loading
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token for loaded tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model_save_path, 
            token=hf_token,
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return
    
    # Create few-shot prompts
    scoring_examples, generation_examples = create_few_shot_prompts(df)
    
    # Initialize results DataFrame
    results = df.copy()
    results['Predicted_Relevance'] = None
    results['Predicted_Fluency'] = None
    results['Predicted_Feasibility'] = None
    results['Predicted_Free_of_vagueness'] = None
    results['Generated_RQ'] = None
    
    # Test scoring
    print("Generating scores...")
    for idx, row in df.iterrows():
        prompt = f"""[INST] Below are example research question scorings:
{chr(10).join([ex['prompt'] + chr(10) + 'Output: ' + ex['target'] for ex in scoring_examples])}

Score the following research question on a scale of 1–5 for each criterion below 
(1 = very poor, 5 = excellent). Provide integer values only.

Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.

Question: {row['RQ']}
Title: {row['Title']}
Output only the JSON scores, e.g., {{"Relevance": 4, "Fluency": 3, "Feasibility": 5, "Free of vagueness": 4}}. [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens (response part)
        response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Parse JSON scores and clamp to 1-5
        try:
            scores = json.loads(generated_text.strip())
            results.at[idx, 'Predicted_Relevance'] = max(1, min(5, int(scores.get('Relevance', 3))))
            results.at[idx, 'Predicted_Fluency'] = max(1, min(5, int(scores.get('Fluency', 3))))
            results.at[idx, 'Predicted_Feasibility'] = max(1, min(5, int(scores.get('Feasibility', 3))))
            results.at[idx, 'Predicted_Free_of_vagueness'] = max(1, min(5, int(scores.get('Free of vagueness', 3))))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing scores for RQ {idx}: {generated_text}")
            results.at[idx, 'Predicted_Relevance'] = 3
            results.at[idx, 'Predicted_Fluency'] = 3
            results.at[idx, 'Predicted_Feasibility'] = 3
            results.at[idx, 'Predicted_Free_of_vagueness'] = 3
    
    # Test RQ generation
    print("Generating improved RQs...")
    for idx, row in df.iterrows():
        comment = row['Comments'] if pd.notna(row['Comments']) else ""
        prompt = f"""[INST] Below are example research question improvements:
{chr(10).join([ex['prompt'] + chr(10) + 'Output: ' + ex['target'] for ex in generation_examples])}

Improve the following research question. Use the title and feedback (if available) as context. Output only the improved research question, ending with a question mark.Improve the following research question. Use the title and feedback (if available) as context. Output only the improved research question, ending with a question mark. 
        The improved RQ should be meeting all of the following criteria: 
        Criteria:
- Relevance: The degree to which the research question aligns with the topic/title.
- Fluency: The grammatical correctness and ease of understanding of the question.
- Feasibility: Whether the research question can realistically be investigated within available resources and constraints.
- Free of vagueness: The extent to which the research question is precise, specific, and avoids ambiguous terms.

Output only ONE improved research question.
- It must be concise, fluent, and end with a "?"

Original Question: {row['RQ']}
Title: {row['Title']}
Feedback: {comment} [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens (response part)
        response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_rq = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Ensure output ends with '?'
        if not generated_rq.endswith('?'):
            generated_rq = generated_rq.rstrip('.') + '?'
        results.at[idx, 'Generated_RQ'] = generated_rq
    
    # Save results to Excel
    output_path = os.path.join(model_save_path, output_excel)
    results.to_excel(output_path, index=False)
    print(f"Test results saved to: {output_path}")
    
    # Print sample results
    print("\nSample test results (first 5 rows):")
    print(results[['RQ', 'Title', 'Relevance', 'Predicted_Relevance', 'Fluency', 'Predicted_Fluency',
                   'Feasibility', 'Predicted_Feasibility', 'Free of vagueness', 'Predicted_Free_of_vagueness',
                   'Improved RQ Suggestion', 'Generated_RQ']].head().to_string())

def main_training_pipeline(excel_file_path='CleanedRQTrain.xlsx', test_only=False):
    """Main training and testing pipeline for Mistral model"""
    
    print("=== Mistral-7B Training and Testing Pipeline ===\n")
    
    os.makedirs('models', exist_ok=True)
    
    df = load_and_preprocess_data(excel_file_path)
    if df is None:
        print("Pipeline aborted due to data loading error.")
        return
    
    if not test_only:
        print("\n=== Training Mistral-7B Model ===")
        mistral_model, mistral_tokenizer, mistral_history = train_mistral_model(df)
        
        training_summary = {
            'total_samples': len(df),
            'mixtral_8x7b': mistral_history
        }
        
        with open('models/mistral_training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print("\n=== Training Complete ===")
        print("Mistral model saved in ./models/mistral_model/")
        print("Training summary saved to ./models/mistral_training_summary.json")
    
    print("\n=== Testing Mistral-7B Model ===")
    test_mistral_model(df)
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    import sys
    test_only = '--test-only' in sys.argv
    main_training_pipeline(test_only=test_only)
