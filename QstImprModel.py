import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import logging
import random

# =========================================================
# T5-Base model for research question improvement
# Uses larger model capacity for higher-quality rewrites
# =========================================================
class T5BaseImprovementModel:
    def __init__(self, model_path='models/t5_base_improvement_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def improve_question(self, question, title="", comments=""):
        """
        Improve a research question using T5-base.
        Prompts are explicitly aligned to the 4 quality criteria:
        Relevance, Fluency, Feasibility, Free of vagueness.
        """
        
        # Construct the prompt (rich instruction with criteria)
        criteria_text = (
            "The improved question must:\n"
            "- Be relevant to the given title.\n"
            "- Be fluent, grammatically correct, and easy to understand.\n"
            "- Be feasible (practical to research within realistic constraints).\n"
            "- Be free of vagueness, precise, and specific.\n"
        )
        
        if comments:
            input_text = (
                f"Improve this research question based on the criteria and feedback.\n"
                f"{criteria_text}\n"
                f"Original Question: {question}\n"
                f"Title: {title}\n"
                f"Feedback: {comments}\n"
                f"Improved Question:"
            )
        else:
            input_text = (
                f"Improve this research question based on the criteria.\n"
                f"{criteria_text}\n"
                f"Original Question: {question}\n"
                f"Title: {title}\n"
                f"Improved Question:"
            )
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].to(self.device)
        attention_mask = input_encoding['attention_mask'].to(self.device)
        
        # Generate improved question
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                min_length=15,
                num_beams=5,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2
            )
        
        improved_question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process for clean output
        improved_question = self._clean_generated_text(improved_question, question)
        
        return improved_question
    
    def _clean_generated_text(self, generated_text, original_question):
        """Remove artifacts and ensure output is a valid RQ ending with '?' """
        
        cleaned = (
            generated_text
            .replace("Improved:", "")
            .replace("Original:", "")
            .replace("Context:", "")
            .strip()
        )
        
        # If too short or identical to original → fallback generic improvement
        if len(cleaned) < 10 or cleaned.lower() == original_question.lower():
            return f"What specific aspects of {original_question.replace('?', '').lower()} should be investigated?"
        
        # Ensure it ends with '?'
        if not cleaned.endswith('?'):
            cleaned += '?'
        
        return cleaned

# =========================================================
# T5-Small model (lighter version, faster but less accurate)
# =========================================================
class T5SmallImprovementModel(T5BaseImprovementModel):
    def __init__(self, model_path='models/t5_small_improvement_model'):
        super().__init__(model_path)
    
    def improve_question(self, question, title="", comments=""):
        """
        Smaller/faster model version for ablation or fallback.
        Same prompt structure, slightly different decoding parameters.
        """
        
        criteria_text = (
            "The improved question must:\n"
            "- Be relevant to the given title.\n"
            "- Be fluent, grammatically correct, and easy to understand.\n"
            "- Be feasible (practical to research within realistic constraints).\n"
            "- Be free of vagueness, precise, and specific.\n"
        )
        
        if comments:
            input_text = (
                f"Improve this research question based on the criteria and feedback.\n"
                f"{criteria_text}\n"
                f"Original Question: {question}\n"
                f"Title: {title}\n"
                f"Feedback: {comments}\n"
                f"Improved Question:"
            )
        else:
            input_text = (
                f"Improve this research question based on the criteria.\n"
                f"{criteria_text}\n"
                f"Original Question: {question}\n"
                f"Title: {title}\n"
                f"Improved Question:"
            )
        
        input_encoding = self.tokenizer(
            input_text,
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
                max_length=256,
                min_length=15,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.85,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        improved_question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self._clean_generated_text(improved_question, question)

# =========================================================
# Good Question Bank: stores high-quality RQs for reference
# =========================================================
class GoodQuestionBank:
    def __init__(self, training_data_path=None):
        self.good_questions = []
        self.bad_to_good_mappings = []
        
        if training_data_path:
            self.load_good_questions(training_data_path)
    
    def load_good_questions(self, excel_path):
        """
        Load good and improved RQs from training data.
        Used for reference-based prompting or template improvements.
        """
        try:
            df = pd.read_excel(excel_path)
            
            # Mark good RQs by flag or improvement
            good_mask = (
                (df['Good RQ'] == 1) |
                (df['Perfect RQ'] == 1) |
                (df['Improved RQ Suggestion'].notna() & (df['Improved RQ Suggestion'] != ''))
            )
            
            good_data = df[good_mask]
            
            for _, row in good_data.iterrows():
                self.good_questions.append({
                    'question': row['RQ'],
                    'title': row['Title'],
                    'relevance': row['Relevance'],
                    'fluency': row['Fluency'],
                    'feasibility': row['Feasibility'],
                    'free_of_vagueness': row['Free of vagueness']
                })
            
            # Map bad → improved examples
            improvement_data = df[df['Improved RQ Suggestion'].notna() & (df['Improved RQ Suggestion'] != '')]
            
            for _, row in improvement_data.iterrows():
                self.bad_to_good_mappings.append({
                    'original': row['RQ'],
                    'improved': row['Improved RQ Suggestion'],
                    'title': row['Title'],
                    'comments': row.get('Comments', '')
                })
            
            print(f"Loaded {len(self.good_questions)} good questions")
            print(f"Loaded {len(self.bad_to_good_mappings)} improvement examples")
        
        except Exception as e:
            print(f"Could not load good questions: {e}")
    
    # Find similar good RQ for inspiration
    def get_similar_good_question(self, target_question, title=""):
        if not self.good_questions:
            return None
        
        target_words = set(target_question.lower().split())
        best_match, best_score = None, 0
        
        for good_q in self.good_questions:
            good_words = set(good_q['question'].lower().split())
            title_words = set(good_q['title'].lower().split())
            
            common_words = len(target_words.intersection(good_words))
            title_similarity = len(set(title.lower().split()).intersection(title_words)) if title else 0
            
            similarity_score = common_words + title_similarity * 0.5
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = good_q
        
        return best_match
    
    # Find similar bad→good mapping
    def get_improvement_example(self, target_question):
        if not self.bad_to_good_mappings:
            return None
        
        target_words = set(target_question.lower().split())
        best_match, best_score = None, 0
        
        for mapping in self.bad_to_good_mappings:
            original_words = set(mapping['original'].lower().split())
            common_words = len(target_words.intersection(original_words))
            
            if common_words > best_score:
                best_score = common_words
                best_match = mapping
        
        return best_match

# =========================================================
# Question Improvement System: integrates both models + bank
# =========================================================
class QuestionImprovementSystem:
    def __init__(self, t5_base_path='models/t5_base_improvement_model',
                 t5_small_path='models/t5_small_improvement_model',
                 training_data_path='CleanedRQTrain.xlsx'):
        print("Loading question improvement system...")
        
        # Load reference question bank
        self.good_question_bank = GoodQuestionBank(training_data_path)
        
        try:
            self.t5_base_improver = T5BaseImprovementModel(t5_base_path)
            print("✓ T5-Base loaded")
        except Exception as e:
            print(f"✗ Failed to load T5-Base: {e}")
            self.t5_base_improver = None
        
        try:
            self.t5_small_improver = T5SmallImprovementModel(t5_small_path)
            print("✓ T5-Small loaded")
        except Exception as e:
            print(f"✗ Failed to load T5-Small: {e}")
            self.t5_small_improver = None
    
    def improve_question(self, question, title="", comments=""):
        """
        Try multiple improvement methods:
        - T5-base
        - T5-small
        - Reference/template from GoodQuestionBank
        """
        results = {}
        
        similar_good = self.good_question_bank.get_similar_good_question(question, title)
        improvement_example = self.good_question_bank.get_improvement_example(question)
        
        # Enhance feedback with example mapping if available
        enhanced_comments = comments
        if improvement_example:
            enhanced_comments += f" Example improvement: '{improvement_example['original']}' became '{improvement_example['improved']}'"
        
        # T5-base
        if self.t5_base_improver:
            try:
                results['t5_base_improved'] = self.t5_base_improver.improve_question(question, title, enhanced_comments)
            except Exception as e:
                print(f"T5-Base failed: {e}")
        
        # T5-small
        if self.t5_small_improver:
            try:
                results['t5_small_improved'] = self.t5_small_improver.improve_question(question, title, enhanced_comments)
            except Exception as e:
                print(f"T5-Small failed: {e}")
        
        # Template fallback using good question
        if similar_good:
            results['template_improved'] = self._create_template_improvement(question, similar_good, title)
        
        return results
    
    def _create_template_improvement(self, question, similar_good, title):
        """
        Simple template-based fallback.
        Uses structure of a known good question to reshape the bad one.
        """
        base_question = question.replace("?", "").lower()
        return f"What specific factors influence {base_question} in the context of {title}?"
