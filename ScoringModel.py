import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
import pandas as pd

class BertMultiTaskModel(nn.Module):
    """BERT-based multi-task model for scoring 4 criteria"""
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        
        # Remove the default classifier
        self.bert.classifier = nn.Identity()
        
        # Custom heads for each scoring criterion
        hidden_size = self.bert.config.hidden_size
        self.relevance_head = nn.Linear(hidden_size, 5)
        self.fluency_head = nn.Linear(hidden_size, 5)
        self.feasibility_head = nn.Linear(hidden_size, 5)
        self.vagueness_head = nn.Linear(hidden_size, 5)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, return_logits=False):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Score each criterion
        relevance_score = torch.argmax(torch.sigmoid(self.relevance_head(pooled_output)), dim=1)
        fluency_score = torch.argmax(torch.sigmoid(self.fluency_head(pooled_output)), dim=1)
        feasibility_score = torch.argmax(torch.sigmoid(self.feasibility_head(pooled_output)), dim=1)
        vagueness_score = torch.argmax(torch.sigmoid(self.vagueness_head(pooled_output)), dim=1)
        
        
        # inside forward()
        relevance_logits = self.relevance_head(pooled_output)
        fluency_logits = self.fluency_head(pooled_output)
        feasibility_logits = self.feasibility_head(pooled_output)
        vagueness_logits = self.vagueness_head(pooled_output)

        """
        # Apply softmax and get class predictions
        relevance_score = torch.argmax(torch.softmax(relevance_logits, dim=1), dim=1) + 1
        fluency_score = torch.argmax(torch.softmax(fluency_logits, dim=1), dim=1) + 1
        feasibility_score = torch.argmax(torch.softmax(feasibility_logits, dim=1), dim=1) + 1
        vagueness_score = torch.argmax(torch.softmax(vagueness_logits, dim=1), dim=1) + 1"""

        
        #return relevance_score, fluency_score, feasibility_score, vagueness_score
        return relevance_logits, fluency_logits, feasibility_logits, vagueness_logits

class T5ScoringDataset(Dataset):
    """Dataset for T5-based scoring"""
    
    def __init__(self, questions, titles, relevance_scores, fluency_scores, 
                 feasibility_scores, vagueness_scores, tokenizer, max_length=512):
        self.questions = questions
        self.titles = titles
        self.relevance_scores = relevance_scores
        self.fluency_scores = fluency_scores
        self.feasibility_scores = feasibility_scores
        self.vagueness_scores = vagueness_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)*4
    
    """def __getitem__(self, idx):
        question = str(self.questions[idx])//4
        title = str(self.titles[idx])%4
        
        # Create scoring tasks for T5
        tasks = [
            (f"score relevance: question: {question} title: {title}", self.relevance_scores[idx]),
            (f"score fluency: {question}", self.fluency_scores[idx]),
            (f"score feasibility: {question}", self.feasibility_scores[idx]),
            (f"score vagueness: {question}", self.vagueness_scores[idx])
        ]
        
        # For this implementation, we'll use the first task (relevance)
        # In practice, you might want to create separate datasets for each task
        #FIX THISSSS MANDIII
        input_text, target_score = tasks[0]
        
        target_text = str(int(target_score))  # Convert score to string
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=32,  # Shorter for score output
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }"""
        
    def __getitem__(self, idx):
        # Determine which question and which criterion this index corresponds to
        question_idx = idx // 4  # Which original question (0, 1, 2, ...)
        criterion_idx = idx % 4  # Which criterion (0=relevance, 1=fluency, 2=feasibility, 3=vagueness)
        
        question = str(self.questions[question_idx])
        title = str(self.titles[question_idx])
        
        # Create different prompts and targets based on the criterion
        if criterion_idx == 0:  # Relevance
            input_text = f"score relevance: question: {question} title: {title}"
            target_score = self.relevance_scores[question_idx]
        elif criterion_idx == 1:  # Fluency
            input_text = f"score fluency: {question}"
            target_score = self.fluency_scores[question_idx]
        elif criterion_idx == 2:  # Feasibility
            input_text = f"score feasibility: {question}"
            target_score = self.feasibility_scores[question_idx]
        else:  # criterion_idx == 3, Free of vagueness
            input_text = f"score vagueness: {question}"
            target_score = self.vagueness_scores[question_idx]
        
        # Convert score to string
        target_text = str(int(target_score))
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=32,  # Short for single digit scores
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class BertScoringModel:
    """BERT-based scoring model wrapper"""
    
    def __init__(self, model_path='models/bert_scoring_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertMultiTaskModel()
        self.model.load_state_dict(torch.load(f"{model_path}/model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def score_question(self, question, title="", link=""):
        """Score a research question using BERT model"""
        
        # Combine question with context
        combined_text = f"Question: {question} Title: {title}"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            relevance, fluency, feasibility, vagueness = self.model(input_ids, attention_mask)
        
        return {
            'method': 'BERT',
            'relevance': round(relevance.item(), 2),
            'fluency': round(fluency.item(), 2),
            'feasibility': round(feasibility.item(), 2),
            'free_of_vagueness': round(vagueness.item(), 2)
            
            #'relevance': int(max(1, min(5, relevance.item()))),
            #'fluency': int(max(1, min(5, fluency.item()))),
            #'feasibility': int(max(1, min(5, feasibility.item()))),
            #'free_of_vagueness': int(max(1, min(5, vagueness.item())))
        }

class T5ScoringModel:
    """T5-based scoring model wrapper"""
    
    def __init__(self, model_path='models/t5_scoring_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def score_question(self, question, title="", link=""):
        """Score a research question using T5 model"""
        
        scores = {}
        
        # Define scoring prompts
        scoring_prompts = {
            'relevance': f"score relevance: question: {question} title: {title}",
            'fluency': f"score fluency: {question}",
            'feasibility': f"score feasibility: {question}",
            'free_of_vagueness': f"score vagueness: {question}"
        }
        
        for criterion, prompt in scoring_prompts.items():
            score = self._get_score_for_prompt(prompt)
            scores[criterion] = score
        
        scores['method'] = 'T5'
        return scores
    
    def _get_score_for_prompt(self, prompt):
        """Get score for a specific prompt"""
        
        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].to(self.device)
        attention_mask = input_encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=1,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode and extract score
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Extract numeric score from generated text
            score = float(generated_text.strip())
            # Ensure score is in valid range
            score = max(1, min(5, score))
        except:
            # Default score if parsing fails
            score = 3.0
        
        return round(score, 2)

class ScoringSystem:
    """Combined scoring system using both BERT and T5"""
    
    def __init__(self, bert_model_path='models/bert_scoring_model', 
                 t5_model_path='models/t5_scoring_model'):
        print("Loading scoring models...")
        try:
            self.bert_scorer = BertScoringModel(bert_model_path)
            print("BERT scoring model loaded")
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            self.bert_scorer = None
        
        try:
            self.t5_scorer = T5ScoringModel(t5_model_path)
            print("T5 scoring model loaded")
        except Exception as e:
            print(f"Failed to load T5 model: {e}")
            self.t5_scorer = None
    
    def score_question(self, question, title="", link=""):
        """Score question using both models and return separate results"""
        
        results = {}
        
        # BERT scoring
        if self.bert_scorer:
            try:
                bert_scores = self.bert_scorer.score_question(question, title, link)
                results['bert_scores'] = bert_scores
            except Exception as e:
                print(f"BERT scoring failed: {e}")
                results['bert_scores'] = None
        
        # T5 scoring
        if self.t5_scorer:
            try:
                t5_scores = self.t5_scorer.score_question(question, title, link)
                results['t5_scores'] = t5_scores
            except Exception as e:
                print(f"T5 scoring failed: {e}")
                results['t5_scores'] = None
        
        return results
    
    def check_needs_improvement(self, scores, threshold=4):
        """Check if question needs improvement based on scores"""
        
        needs_improvement = {'bert': False, 't5': False}
        
        if scores.get('bert_scores'):
            bert_scores = scores['bert_scores']
            bert_low_scores = [
                score for key, score in bert_scores.items() 
                if key != 'method' and score < threshold
            ]
            needs_improvement['bert'] = len(bert_low_scores) > 0
        
        if scores.get('t5_scores'):
            t5_scores = scores['t5_scores']
            t5_low_scores = [
                score for key, score in t5_scores.items() 
                if key != 'method' and score < threshold
            ]
            needs_improvement['t5'] = len(t5_low_scores) > 0
        
        return needs_improvement
    
    def print_score_breakdown(self, question, scores, needs_improvement):
        """Print detailed score breakdown"""
        
        print(f"\n=== SCORE BREAKDOWN ===")
        print(f"Question: {question}")
        print("-" * 50)
        
        # BERT scores
        if scores.get('bert_scores'):
            bert_scores = scores['bert_scores']
            print(f"BERT Results:")
            print(f"  Relevance: {bert_scores['relevance']}/5")
            print(f"  Fluency: {bert_scores['fluency']}/5")
            print(f"  Feasibility: {bert_scores['feasibility']}/5")
            print(f"  Free of Vagueness: {bert_scores['free_of_vagueness']}/5")
            print(f"  Needs Improvement: {'Yes' if needs_improvement['bert'] else 'No'}")
        
        print()
        
        # T5 scores
        if scores.get('t5_scores'):
            t5_scores = scores['t5_scores']
            print(f"T5 Results:")
            print(f"  Relevance: {t5_scores['relevance']}/5")
            print(f"  Fluency: {t5_scores['fluency']}/5")
            print(f"  Feasibility: {t5_scores['feasibility']}/5")
            print(f"  Free of Vagueness: {t5_scores['free_of_vagueness']}/5")
            print(f"  Needs Improvement: {'Yes' if needs_improvement['t5'] else 'No'}")
        
        print("-" * 50)