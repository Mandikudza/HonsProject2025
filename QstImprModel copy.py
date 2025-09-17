import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import logging
import random

class T5BaseImprovementModel:
    """T5-Base model for question improvement"""
    
    def __init__(self, model_path='models/t5_base_improvement_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def improve_question(self, question, title="", comments=""):
        """Improve a research question using T5-base model with better prompting"""
        
        # Create more specific input prompt
        if comments:
            input_text = f"Improve this research question based on feedback. Original: {question} Context: {title} Feedback: {comments} Improved:"
        else:
            input_text = f"Improve this research question to be more specific and clear. Original: {question} Context: {title} Improved:"
        
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
                min_length=20,
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
        
        # Clean up the output - remove any prompting artifacts
        improved_question = self._clean_generated_text(improved_question, question)
        
        return improved_question
    
    def _clean_generated_text(self, generated_text, original_question):
        """Clean up generated text to extract just the improved question"""
        
        # Remove common prompt artifacts
        cleaned = generated_text.replace("Improved:", "").strip()
        cleaned = cleaned.replace("Original:", "").strip()
        cleaned = cleaned.replace("Context:", "").strip()
        
        # If the generated text is too similar to input or too short, return a basic improvement
        if len(cleaned) < 10 or cleaned.lower() == original_question.lower():
            return f"What specific aspects of {original_question.replace('?', '').lower()} should be investigated?"
        
        # Ensure it ends with a question mark
        if not cleaned.endswith('?'):
            cleaned += '?'
        
        return cleaned

class T5SmallImprovementModel:
    """T5-Small model for question improvement"""
    
    def __init__(self, model_path='models/t5_small_improvement_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def improve_question(self, question, title="", comments=""):
        """Improve a research question using T5-small model with better prompting"""
        
        # Create more specific input prompt
        if comments:
            input_text = f"Improve this research question based on feedback. Original: {question} Context: {title} Feedback: {comments} Improved:"
        else:
            input_text = f"Improve this research question to be more specific and clear. Original: {question} Context: {title} Improved:"
        
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
                min_length=20,
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
        
        # Clean up the output
        improved_question = self._clean_generated_text(improved_question, question)
        
        return improved_question
    
    def _clean_generated_text(self, generated_text, original_question):
        """Clean up generated text to extract just the improved question"""
        
        # Remove common prompt artifacts
        cleaned = generated_text.replace("Improved:", "").strip()
        cleaned = cleaned.replace("Original:", "").strip()
        cleaned = cleaned.replace("Context:", "").strip()
        
        # If the generated text is too similar to input or too short, return a basic improvement
        if len(cleaned) < 10 or cleaned.lower() == original_question.lower():
            return f"How does {original_question.replace('?', '').lower()} impact specific outcomes?"
        
        # Ensure it ends with a question mark
        if not cleaned.endswith('?'):
            cleaned += '?'
        
        return cleaned

class GoodQuestionBank:
    """Bank of good research questions for reference-based improvement"""
    
    def __init__(self, training_data_path=None):
        self.good_questions = []
        self.bad_to_good_mappings = []
        
        if training_data_path:
            self.load_good_questions(training_data_path)
    
    def load_good_questions(self, excel_path):
        """Load good questions from training data"""
        try:
            df = pd.read_excel(excel_path)
            
            # Extract good questions (those with high scores or marked as good/perfect)
            good_mask = (
                (df['Good RQ'] == 1) | 
                (df['Perfect RQ'] == 1) | 
                (df['Improved RQ Suggestion'].notna() & (df['Improved RQ Suggestion'] != ''))
            )
            
            good_data = df[good_mask]
            
            # Store good questions with their scores
            for _, row in good_data.iterrows():
                self.good_questions.append({
                    'question': row['RQ'],
                    'title': row['Title'],
                    'relevance': row['Relevance'],
                    'fluency': row['Fluency'],
                    'feasibility': row['Feasibility'],
                    'free_of_vagueness': row['Free of vagueness']
                })
            
            # Store bad to good mappings
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
    
    def get_similar_good_question(self, target_question, title=""):
        """Find a similar good question for reference"""
        if not self.good_questions:
            return None
        
        # Simple similarity based on common words QstImprModel.
        target_words = set(target_question.lower().split())
        
        best_match = None
        best_score = 0
        
        for good_q in self.good_questions:
            good_words = set(good_q['question'].lower().split())
            title_words = set(good_q['title'].lower().split())
            
            # Calculate similarity score
            common_words = len(target_words.intersection(good_words))
            title_similarity = len(set(title.lower().split()).intersection(title_words)) if title else 0
            
            similarity_score = common_words + title_similarity * 0.5
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = good_q
        
        return best_match
    
    def get_improvement_example(self, target_question):
        """Get a similar improvement example"""
        if not self.bad_to_good_mappings:
            return None
        
        target_words = set(target_question.lower().split())
        
        best_match = None
        best_score = 0
        
        for mapping in self.bad_to_good_mappings:
            original_words = set(mapping['original'].lower().split())
            common_words = len(target_words.intersection(original_words))
            
            if common_words > best_score:
                best_score = common_words
                best_match = mapping
        
        return best_match

class QuestionImprovementSystem:
    """Enhanced question improvement system using both T5 models and good question bank"""
    
    def __init__(self, t5_base_path='models/t5_base_improvement_model',
                 t5_small_path='models/t5_small_improvement_model',
                 training_data_path='CleanedRQTrain.xlsx'):
        print("Loading question improvement models...")
        
        # Load good question bank
        self.good_question_bank = GoodQuestionBank(training_data_path)
        
        try:
            self.t5_base_improver = T5BaseImprovementModel(t5_base_path)
            print(" T5-Base improvement model loaded")
        except Exception as e:
            print(f" Failed to load T5-Base model: {e}")
            self.t5_base_improver = None
        
        try:
            self.t5_small_improver = T5SmallImprovementModel(t5_small_path)
            print(" T5-Small improvement model loaded")
        except Exception as e:
            print(f" Failed to load T5-Small model: {e}")
            self.t5_small_improver = None
    
    def improve_question(self, question, title="", comments=""):
        """Improve question using both models and return separate results"""
        
        results = {}
        
        # Get reference from good question bank
        similar_good = self.good_question_bank.get_similar_good_question(question, title)
        improvement_example = self.good_question_bank.get_improvement_example(question)
        
        # Enhance comments with reference examples
        enhanced_comments = comments
        if improvement_example:
            enhanced_comments += f" Example improvement: '{improvement_example['original']}' became '{improvement_example['improved']}'"
        
        # T5-Base improvement
        if self.t5_base_improver:
            try:
                t5_base_improved = self.t5_base_improver.improve_question(question, title, enhanced_comments)
                results['t5_base_improved'] = t5_base_improved
            except Exception as e:
                print(f"T5-Base improvement failed: {e}")
                results['t5_base_improved'] = None
        
        # T5-Small improvement
        if self.t5_small_improver:
            try:
                t5_small_improved = self.t5_small_improver.improve_question(question, title, enhanced_comments)
                results['t5_small_improved'] = t5_small_improved
            except Exception as e:
                print(f"T5-Small improvement failed: {e}")
                results['t5_small_improved'] = None
        
        # Template-based improvement as fallback
        if similar_good:
            template_improved = self._create_template_improvement(question, similar_good, title)
            results['template_improved'] = template_improved
        
        return results
    
    def _create_template_improvement(self, question, similar_good, title):
        """Create template-based improvement using a similar good question"""
        
        # Extract key patterns from the good question
        good_question = similar_good['question']
        
        # Simple template matching (Common feedback factors)
        if "what" in question.lower() and "how" in good_question.lower():
            # Convert "what" to "how"
            improved = question.replace("What", "How").replace("what", "how")
        elif "how" in question.lower() and "what specific" in good_question.lower():
            # Add specificity
            improved = question.replace("How", "What specific aspects of how")
        else:
            # Add specificity and context
            base_question = question.replace("?", "")
            improved = f"What specific factors influence {base_question.lower()} in the context of {title}?"
        
        return improved
    
    def print_improvement_results(self, original_question, improvement_results):
        """Print improvement results from all methods"""
        
        print(f"\n=== QUESTION IMPROVEMENT RESULTS ===")
        print(f"Original Question: {original_question}")
        print("-" * 60)
        
        if improvement_results.get('t5_base_improved'):
            print(f"T5-Base Improved:")
            print(f"  {improvement_results['t5_base_improved']}")
            print()
        
        if improvement_results.get('t5_small_improved'):
            print(f"T5-Small Improved:")
            print(f"  {improvement_results['t5_small_improved']}")
            print()
        
        if improvement_results.get('template_improved'):
            print(f"Template-Based Improved:")
            print(f"  {improvement_results['template_improved']}")
            print()
        
        print("-" * 60)
    
    def get_best_improvement(self, improvement_results):
        """Select the best improvement with better logic"""
        
        # Prioritize non-empty results
        options = []
        
        if improvement_results.get('t5_base_improved') and len(improvement_results['t5_base_improved']) > 10:
            options.append(('t5_base', improvement_results['t5_base_improved']))
        
        if improvement_results.get('t5_small_improved') and len(improvement_results['t5_small_improved']) > 10:
            options.append(('t5_small', improvement_results['t5_small_improved']))
        
        if improvement_results.get('template_improved') and len(improvement_results['template_improved']) > 10:
            options.append(('template', improvement_results['template_improved']))
        
        # Return the first valid option, preferring T5-base
        if options:
            return options[0][1]
        else:
            return None
    
    def improve_with_feedback(self, question, title="", low_score_areas=None):
        """Improve question with specific feedback based on low scoring areas"""
        
        # Generate specific feedback based on low scoring areas
        feedback_comments = []
        
        if low_score_areas:
            if 'relevance' in low_score_areas:
                feedback_comments.append("make more relevant to the title and context")
            if 'fluency' in low_score_areas:
                feedback_comments.append("improve clarity and readability")
            if 'feasibility' in low_score_areas:
                feedback_comments.append("make more practical and achievable")
            if 'free_of_vagueness' in low_score_areas:
                feedback_comments.append("be more specific and precise, avoid vague terms")
        
        comments = "; ".join(feedback_comments)
        
        return self.improve_question(question, title, comments)
    
    def batch_improve_questions(self, questions_data, output_file=None):
        """Improve multiple questions and save results"""
        
        results = []
        
        for i, data in enumerate(questions_data):
            question = data.get('question', '')
            title = data.get('title', '')
            comments = data.get('comments', '')
            
            print(f"Improving question {i+1}/{len(questions_data)}")
            
            improvement_results = self.improve_question(question, title, comments)
            
            result = {
                'original_question': question,
                'title': title,
                'comments': comments,
                't5_base_improved': improvement_results.get('t5_base_improved'),
                't5_small_improved': improvement_results.get('t5_small_improved'),
                'template_improved': improvement_results.get('template_improved'),
                'best_improvement': self.get_best_improvement(improvement_results)
            }
            
            results.append(result)
        
        # Save to Excel if output file specified
        if output_file:
            results_df = pd.DataFrame(results)
            results_df.to_excel(output_file, index=False)
            print(f"Improvement results saved to: {output_file}")
        
        return results

class AdvancedImprovementFeatures:
    """Advanced features for question improvement"""
    
    @staticmethod
    def analyze_improvement_quality(original, improved):
        """Analyze the quality of improvement"""
        if not improved or improved.strip() == "" or len(improved) < 5:
            return {
                'quality_score': 0,
                'feedback': ["No valid improvement generated"],
                'improvement_type': 'None'
            }
        
        if original.lower().strip() == improved.lower().strip():
            return {
                'quality_score': 0,
                'feedback': ["No changes made"],
                'improvement_type': 'None'
            }
        
        # Basic quality checks
        quality_score = 0
        feedback = []
        
        # Length and detail check
        if len(improved) > len(original) * 1.1:  # At least 10% longer
            quality_score += 1
            feedback.append("Added detail")
        
        # Word count check
        if len(improved.split()) > len(original.split()):
            quality_score += 1
            feedback.append("More comprehensive")
        
        # Question structure check
        if improved.endswith('?'):
            quality_score += 1
            feedback.append("Proper question format")
        
        # Specificity check (contains specific terms)
        specific_terms = ['specific', 'particular', 'exactly', 'precisely', 'which', 'what type', 'how much', 'to what extent']
        if any(term in improved.lower() for term in specific_terms):
            quality_score += 1
            feedback.append("More specific")
        
        # Avoid repetitive or template-like responses
        if "improve" in improved.lower() or "original:" in improved.lower():
            quality_score -= 1
            feedback.append("Contains template artifacts")
        
        return {
            'quality_score': max(0, quality_score),
            'feedback': feedback,
            'improvement_type': 'Enhanced' if quality_score >= 3 else 'Basic' if quality_score > 0 else 'Poor'
        }
    
    @staticmethod
    def suggest_improvement_strategy(scores):
        """Suggest improvement strategy based on scores"""
        
        strategies = []
        
        if scores.get('relevance', 5) < 4:
            strategies.append("Focus on connecting the question more directly to the research context")
        
        if scores.get('fluency', 5) < 4:
            strategies.append("Improve sentence structure and clarity")
        
        if scores.get('feasibility', 5) < 4:
            strategies.append("Make the research scope more achievable and practical")
        
        if scores.get('free_of_vagueness', 5) < 4:
            strategies.append("Add specific details and remove ambiguous terms")
        
        return strategies
    
    @staticmethod
    def compare_improvements(t5_base_result, t5_small_result, original):
        """Compare improvements from both models"""
        
        comparison = {
            'original': original,
            't5_base': t5_base_result if t5_base_result else '',
            't5_small': t5_small_result if t5_small_result else '',
            't5_base_analysis': {'quality_score': 0, 'feedback': ['No improvement'], 'improvement_type': 'None'},
            't5_small_analysis': {'quality_score': 0, 'feedback': ['No improvement'], 'improvement_type': 'None'},
            'recommendation': None
        }
        
        # Analyze each improvement
        if t5_base_result:
            try:
                comparison['t5_base_analysis'] = AdvancedImprovementFeatures.analyze_improvement_quality(
                    original, t5_base_result
                )
            except Exception as e:
                logging.error(f"Error analyzing T5-base improvement: {e}")
        
        if t5_small_result:
            try:
                comparison['t5_small_analysis'] = AdvancedImprovementFeatures.analyze_improvement_quality(
                    original, t5_small_result
                )
            except Exception as e:
                logging.error(f"Error analyzing T5-small improvement: {e}")
        
        # Recommend best option based on quality scores
        base_quality = comparison['t5_base_analysis']['quality_score']
        small_quality = comparison['t5_small_analysis']['quality_score']
        
        if base_quality > small_quality and base_quality > 0:
            comparison['recommendation'] = 't5_base'
        elif small_quality > base_quality and small_quality > 0:
            comparison['recommendation'] = 't5_small'
        elif t5_base_result and len(t5_base_result) > 10:
            comparison['recommendation'] = 't5_base'  # Default to base if both are poor
        elif t5_small_result and len(t5_small_result) > 10:
            comparison['recommendation'] = 't5_small'
        else:
            comparison['recommendation'] = None
        
        return comparison