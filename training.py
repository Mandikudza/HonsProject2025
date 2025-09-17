import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json
import os
from charts import plot_training_history_matplotlib, generate_training_history_chartjs

class RQScoringDataset(Dataset):
    """Dataset for research question scoring with multiple criteria"""
    
    def __init__(self, questions, titles, links, relevance_scores, fluency_scores, 
                 feasibility_scores, vagueness_scores, tokenizer, max_length=512):
        self.questions = questions
        self.titles = titles
        self.links = links
        self.relevance_scores = relevance_scores
        self.fluency_scores = fluency_scores
        self.feasibility_scores = feasibility_scores
        self.vagueness_scores = vagueness_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        title = str(self.titles[idx])
        
        combined_text = f"Question: {question} Title: {title}"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relevance_score': torch.tensor(self.relevance_scores[idx], dtype=torch.float),
            'fluency_score': torch.tensor(self.fluency_scores[idx], dtype=torch.float),
            'feasibility_score': torch.tensor(self.feasibility_scores[idx], dtype=torch.float),
            'vagueness_score': torch.tensor(self.vagueness_scores[idx], dtype=torch.float)
        }

class RQImprovementDataset(Dataset):
    """Dataset for research question improvement"""
    
    def __init__(self, original_questions, improved_questions, titles, comments, tokenizer, max_length=512):
        self.original_questions = original_questions
        self.improved_questions = improved_questions
        self.titles = titles
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.original_questions)
    
    def __getitem__(self, idx):
        original = str(self.original_questions[idx])
        improved = str(self.improved_questions[idx])
        title = str(self.titles[idx])
        comment = str(self.comments[idx]) if pd.notna(self.comments[idx]) else ""
        
        input_text = f"improve research question: {original} context: {title} feedback: {comment}. Output only the improved research question, no explanations or comments."
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            improved,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

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
    
    # Normalize column names (strip whitespace, convert to consistent case)
    df.columns = [col.strip().replace('\xa0', ' ') for col in df.columns]
    print("Normalized columns:", df.columns.tolist())
    
    # Verify required columns
    required_columns = ['RQ', 'Title', 'Relevance', 'Fluency', 'Feasibility', 'Free of vagueness', 
                       'Bad RQ', 'Good RQ', 'Perfect RQ', 'Improved RQ Suggestion', 'Comments']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return None
    
    # Log raw data summary
    print("\nRaw data summary:")
    print(f"Rows with Bad RQ == 'y': {len(df[df['Bad RQ'] == 'y'])}")
    print(f"Rows with Bad RQ == 'n': {len(df[df['Bad RQ'] == 'n'])}")
    print(f"Rows with Good RQ == 'y': {len(df[df['Good RQ'] == 'y'])}")
    print(f"Rows with Perfect RQ == 'y': {len(df[df['Perfect RQ'] == 'y'])}")
    print(f"Rows with non-null Improved RQ Suggestion: {len(df[df['Improved RQ Suggestion'].notna()])}")
    
    # Convert 'y'/'n' to 1/0 for Bad RQ, Good RQ, Perfect RQ
    for col in ['Bad RQ', 'Good RQ', 'Perfect RQ']:
        print(f"\nUnique values in {col} before conversion:", df[col].unique().tolist())
        df[col] = df[col].map({'y': 1, 'n': 0}).fillna(0).astype(int)
        print(f"Unique values in {col} after conversion:", df[col].unique().tolist())
        non_numeric = df[col][pd.to_numeric(df[col], errors='coerce').isna()]
        if not non_numeric.empty:
            print(f"Non-numeric values in {col} after mapping:", non_numeric.tolist())
    
    # Clean and validate data
    initial_len = len(df)
    df = df.dropna(subset=['RQ', 'Title'])
    print(f"\nAfter dropping missing RQ/Title: {len(df)} rows (dropped {initial_len - len(df)})")
    
    score_columns = ['Relevance', 'Fluency', 'Feasibility', 'Free of vagueness']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].clip(1, 5)
    
    initial_len = len(df)
    df = df.dropna(subset=score_columns)
    print(f"After dropping missing scores: {len(df)} rows (dropped {initial_len - len(df)})")
    
    improved_count = df['Improved RQ Suggestion'].notna().sum()
    print(f"Questions with improvements: {improved_count}")
    
    # Log filtered data summary
    print("\nFiltered data summary:")
    print(f"Rows with Bad RQ == 1: {len(df[df['Bad RQ'] == 1])}")
    print(f"Rows with Bad RQ == 0: {len(df[df['Bad RQ'] == 0])}")
    print(f"Rows with Good RQ == 1: {len(df[df['Good RQ'] == 1])}")
    print(f"Rows with Perfect RQ == 1: {len(df[df['Perfect RQ'] == 1])}")
    print(f"Rows with non-null Improved RQ Suggestion: {len(df[df['Improved RQ Suggestion'].notna()])}")
    
    return df

def train_bert_scoring_model(df, model_save_path='models/bert_scoring_model'):
    """Train BERT-based scoring model"""
    
    print("=== Training BERT Scoring Model ===")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    questions = df['RQ'].tolist()
    titles = df['Title'].tolist()
    links = df['link'].fillna('').tolist()
    relevance = df['Relevance'].tolist()
    fluency = df['Fluency'].tolist()
    feasibility = df['Feasibility'].tolist()
    vagueness = df['Free of vagueness'].tolist()
    
    all_scores = relevance + fluency + feasibility + vagueness
    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)
    print(f"Score mean: {score_mean}, std: {score_std}")
    
    norm_relevance = [(s - score_mean) / score_std for s in relevance]
    norm_fluency = [(s - score_mean) / score_std for s in fluency]
    norm_feasibility = [(s - score_mean) / score_std for s in feasibility]
    norm_vagueness = [(s - score_mean) / score_std for s in vagueness]
    
    train_q, test_q, train_t, test_t, train_l, test_l, train_rel, test_rel, \
    train_flu, test_flu, train_fea, test_fea, train_vag, test_vag = train_test_split(
        questions, titles, links, norm_relevance, norm_fluency, norm_feasibility, norm_vagueness,
        test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_q)}")
    print(f"Test samples: {len(test_q)}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = RQScoringDataset(train_q, train_t, train_l, train_rel, train_flu, train_fea, train_vag, tokenizer)
    test_dataset = RQScoringDataset(test_q, test_t, test_l, test_rel, test_flu, test_fea, test_vag, tokenizer)
    
    from ScoringModel import BertMultiTaskModel
    model = BertMultiTaskModel()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    num_epochs = 10
    train_losses = []
    test_losses = []
   
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            #rel_pred, flu_pred, fea_pred, vag_pred = model(input_ids, attention_mask)
            rel_logits, flu_logits, fea_logits, vag_logits = model(input_ids, attention_mask)
            """
            rel_loss = criterion(rel_pred.squeeze(), batch['relevance_score'].to(device))
            flu_loss = criterion(flu_pred.squeeze(), batch['fluency_score'].to(device))
            fea_loss = criterion(fea_pred.squeeze(), batch['feasibility_score'].to(device))
            vag_loss = criterion(vag_pred.squeeze(), batch['vagueness_score'].to(device))"""
            
            # Shift labels from 1–5 → 0–4 for CE
            rel_loss = criterion(rel_logits, (batch['relevance_score'].to(device) - 1).long())
            flu_loss = criterion(flu_logits, (batch['fluency_score'].to(device) - 1).long())
            fea_loss = criterion(fea_logits, (batch['feasibility_score'].to(device) - 1).long())
            vag_loss = criterion(vag_logits, (batch['vagueness_score'].to(device) - 1).long())
                
            total_loss = rel_loss + flu_loss + fea_loss + vag_loss
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
        
        model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                rel_pred, flu_pred, fea_pred, vag_pred = model(input_ids, attention_mask)
                
                rel_loss = criterion(rel_pred.squeeze(), batch['relevance_score'].to(device))
                flu_loss = criterion(flu_pred.squeeze(), batch['fluency_score'].to(device))
                fea_loss = criterion(fea_pred.squeeze(), batch['feasibility_score'].to(device))
                vag_loss = criterion(vag_pred.squeeze(), batch['vagueness_score'].to(device))
                
                total_test_loss += (rel_loss + flu_loss + fea_loss + vag_loss).item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    torch.save(model.state_dict(), f"{model_save_path}/model.pth")
    tokenizer.save_pretrained(model_save_path)
    
    np.save(f"{model_save_path}/score_stats.npy", np.array([score_mean, score_std]))
    
    training_history = {
        'model_type': 'bert_scoring',
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_size': len(train_q),
        'test_size': len(test_q)
    }
    
    with open(f"{model_save_path}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plot_training_history_matplotlib(training_history, 'BERT Scoring Model', f"{model_save_path}/training_history.png")
    generate_training_history_chartjs(training_history, 'BERT Scoring Model', f"{model_save_path}/training_history_chart.json")
    
    print(f"BERT scoring model saved to: {model_save_path}")
    return model, tokenizer, training_history

def train_t5_scoring_model(df, model_save_path='models/t5_scoring_model'):
    """Train T5-based scoring model"""
    
    print("=== Training T5 Scoring Model ===")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    questions = df['RQ'].tolist()
    titles = df['Title'].tolist()
    relevance = df['Relevance'].tolist()
    fluency = df['Fluency'].tolist()
    feasibility = df['Feasibility'].tolist()
    vagueness = df['Free of vagueness'].tolist()
    
    train_q, test_q, train_t, test_t, train_rel, test_rel, \
    train_flu, test_flu, train_fea, test_fea, train_vag, test_vag = train_test_split(
        questions, titles, relevance, fluency, feasibility, vagueness,
        test_size=0.2, random_state=42
    )
    
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    
    from ScoringModel import T5ScoringDataset
    train_dataset = T5ScoringDataset(train_q, train_t, train_rel, train_flu, train_fea, train_vag, tokenizer)
    test_dataset = T5ScoringDataset(test_q, test_t, test_rel, test_flu, test_fea, test_vag, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{model_save_path}/logs',
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    training_result = trainer.train()
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    training_history = {
        'model_type': 't5_scoring',
        'train_losses': train_losses,
        'test_losses': eval_losses,
        'train_size': len(train_q),
        'test_size': len(test_q)
    }
    
    with open(f"{model_save_path}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plot_training_history_matplotlib(training_history, 'T5 Scoring Model', f"{model_save_path}/training_history.png")
    generate_training_history_chartjs(training_history, 'T5 Scoring Model', f"{model_save_path}/training_history_chart.json")
    
    print(f"T5 scoring model saved to: {model_save_path}")
    return model, tokenizer, training_history

def train_t5_base_improvement_model(df, model_save_path='models/t5_base_improvement_model'):
    """Train T5-base for question improvement"""
    
    print("=== Training T5-Base Improvement Model ===")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    # Step 1: Collect bad RQs with existing improvements
    df_bad = df[(df['Bad RQ'] == 1) & df['Improved RQ Suggestion'].notna()].copy()
    print(f"Bad RQs with improvements: {len(df_bad)}")
    if len(df_bad) == 0:
        print("No bad RQs with improvements found. Check 'Bad RQ' and 'Improved RQ Suggestion' columns.")
    
    # Step 2: Collect good/perfect RQs and augment with self-supervision
    df_good = df[(df['Bad RQ'] == 0) & ((df['Good RQ'] == 1) | (df['Perfect RQ'] == 1))].copy()
    print(f"Good/Perfect RQs before self-supervision: {len(df_good)}")
    df_good.loc[df_good['Improved RQ Suggestion'].isna(), 'Improved RQ Suggestion'] = df_good.loc[df_good['Improved RQ Suggestion'].isna(), 'RQ']
    print(f"Good/Perfect RQs (self-supervised): {len(df_good)}")
    if len(df_good) == 0:
        print("No good/perfect RQs found. Check 'Bad RQ', 'Good RQ', and 'Perfect RQ' columns.")
    
    # Step 3: Combine bad (with improvements) and good (self-supervised)
    df_improved = pd.concat([df_bad, df_good], ignore_index=True)
    print(f"Total improvement examples: {len(df_improved)}")
    
    if len(df_improved) == 0:
        print("ERROR: No data available for training T5-base improvement model. Skipping training.")
        return None, None, None
    
    # Prepare data
    original_questions = df_improved['RQ'].tolist()
    improved_questions = df_improved['Improved RQ Suggestion'].tolist()
    titles = df_improved['Title'].tolist()
    comments = df_improved['Comments'].fillna('').tolist()
    
    # 80/20 split
    train_orig, test_orig, train_imp, test_imp, train_titles, test_titles, train_comments, test_comments = train_test_split(
        original_questions, improved_questions, titles, comments,
        test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_orig)}")
    print(f"Test samples: {len(test_orig)}")
    
    # Initialize T5-base
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    
    # Create datasets
    train_dataset = RQImprovementDataset(train_orig, train_imp, train_titles, train_comments, tokenizer)
    test_dataset = RQImprovementDataset(test_orig, test_imp, test_titles, test_comments, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{model_save_path}/logs',
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    training_result = trainer.train()
    
    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Extract losses
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    training_history = {
        'model_type': 't5_base_improvement',
        'train_losses': train_losses,
        'test_losses': eval_losses,
        'train_size': len(train_orig),
        'test_size': len(test_orig)
    }
    
    with open(f"{model_save_path}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plot_training_history_matplotlib(training_history, 'T5-Base Improvement Model', f"{model_save_path}/training_history.png")
    generate_training_history_chartjs(training_history, 'T5-Base Improvement Model', f"{model_save_path}/training_history_chart.json")
    
    print(f"T5-base improvement model saved to: {model_save_path}")
    return model, tokenizer, training_history

def train_t5_small_improvement_model(df, model_save_path='models/t5_small_improvement_model'):
    """Train T5-small for question improvement"""
    
    print("=== Training T5-Small Improvement Model ===")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    # Step 1: Collect bad RQs with existing improvements
    df_bad = df[(df['Bad RQ'] == 1) & df['Improved RQ Suggestion'].notna()].copy()
    print(f"Bad RQs with improvements: {len(df_bad)}")
    if len(df_bad) == 0:
        print("No bad RQs with improvements found. Check 'Bad RQ' and 'Improved RQ Suggestion' columns.")
    
    # Step 2: Collect good/perfect RQs and augment with self-supervision
    df_good = df[(df['Bad RQ'] == 0) & ((df['Good RQ'] == 1) | (df['Perfect RQ'] == 1))].copy()
    print(f"Good/Perfect RQs before self-supervision: {len(df_good)}")
    df_good.loc[df_good['Improved RQ Suggestion'].isna(), 'Improved RQ Suggestion'] = df_good.loc[df_good['Improved RQ Suggestion'].isna(), 'RQ']
    print(f"Good/Perfect RQs (self-supervised): {len(df_good)}")
    if len(df_good) == 0:
        print("No good/perfect RQs found. Check 'Bad RQ', 'Good RQ', and 'Perfect RQ' columns.")
    
    # Step 3: Combine bad (with improvements) and good (self-supervised)
    df_improved = pd.concat([df_bad, df_good], ignore_index=True)
    print(f"Total improvement examples: {len(df_improved)}")
    
    if len(df_improved) == 0:
        print("ERROR: No data available for training T5-small improvement model. Skipping training.")
        return None, None, None
    
    # Prepare data
    original_questions = df_improved['RQ'].tolist()
    improved_questions = df_improved['Improved RQ Suggestion'].tolist()
    titles = df_improved['Title'].tolist()
    comments = df_improved['Comments'].fillna('').tolist()
    
    # 80/20 split
    train_orig, test_orig, train_imp, test_imp, train_titles, test_titles, train_comments, test_comments = train_test_split(
        original_questions, improved_questions, titles, comments,
        test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_orig)}")
    print(f"Test samples: {len(test_orig)}")
    
    # Initialize T5-small
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
    
    # Create datasets
    train_dataset = RQImprovementDataset(train_orig, train_imp, train_titles, train_comments, tokenizer)
    test_dataset = RQImprovementDataset(test_orig, test_imp, test_titles, test_comments, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{model_save_path}/logs',
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    training_result = trainer.train()
    
    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Extract losses
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    training_history = {
        'model_type': 't5_small_improvement',
        'train_losses': train_losses,
        'test_losses': eval_losses,
        'train_size': len(train_orig),
        'test_size': len(test_orig)
    }
    
    with open(f"{model_save_path}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plot_training_history_matplotlib(training_history, 'T5-Small Improvement Model', f"{model_save_path}/training_history.png")
    generate_training_history_chartjs(training_history, 'T5-Small Improvement Model', f"{model_save_path}/training_history_chart.json")
    
    print(f"T5-small improvement model saved to: {model_save_path}")
    return model, tokenizer, training_history

def main_training_pipeline(excel_file_path='CleanedRQTrain.xlsx'):
    """Main training pipeline for all models"""
    
    print("=== Research Question Models Training Pipeline ===\n")
    
    os.makedirs('models', exist_ok=True)
    
    df = load_and_preprocess_data(excel_file_path)
    if df is None:
        print("Training aborted due to data loading error.")
        return
    
    print("\n=== Training All Models ===")
    
    bert_model, bert_tokenizer, bert_history = train_bert_scoring_model(df)
    
    t5_scoring_model, t5_scoring_tokenizer, t5_scoring_history = train_t5_scoring_model(df)
    
    t5_base_model, t5_base_tokenizer, t5_base_history = train_t5_base_improvement_model(df)
    
    t5_small_model, t5_small_tokenizer, t5_small_history = train_t5_small_improvement_model(df)
    
    training_summary = {
        'total_samples': len(df),
        'bert_scoring': bert_history,
        't5_scoring': t5_scoring_history,
        't5_base_improvement': t5_base_history,
        't5_small_improvement': t5_small_history
    }
    
    with open('models/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("\n=== Training Complete ===")
    print("All models saved in ./models/ directory")
    print("Training summary saved to ./models/training_summary.json")

if __name__ == "__main__":
    main_training_pipeline()