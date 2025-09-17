import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
from transformers import BertTokenizer, BertModel
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RQComparator:
    """
    Class to compare research questions using ROUGE-L (Lin 2004) and BERTScore F1 (Zhang et al. 2019)
    Compares: Original RQ, Improved RQ Suggestion, T5-Base Generated, T5-Small Generated
    """
    
    def __init__(self):
        print("Initializing RQ Comparison System...")
        print("- ROUGE-L implementation based on Lin (2004)")
        print("- BERTScore F1 implementation based on Zhang et al. (2019)")
        
        # Initialize ROUGE scorer for ROUGE-L (Lin 2004)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Initialize BERT model for BERTScore (Zhang et al. 2019)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"- Using device: {self.device}")
        
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("BERT model loaded for BERTScore calculations")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
        
        print("RQ Comparison System ready :)\n")
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> Dict:
        """
        Calculate ROUGE-L score between reference and candidate (Lin 2004)
        
        Args:
            reference: Ground truth/reference text
            candidate: Generated/candidate text to compare
            
        Returns:
            Dict with rouge_l_precision, rouge_l_recall, rouge_l_f1
        """
        try:
            if not reference or not candidate:
                return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
            
            scores = self.rouge_scorer.score(reference.strip(), candidate.strip())
            rouge_l = scores['rougeL']
            
            return {
                'rouge_l_precision': rouge_l.precision,
                'rouge_l_recall': rouge_l.recall,
                'rouge_l_f1': rouge_l.fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE-L: {e}")
            return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
    
    def calculate_bertscore_f1(self, reference: str, candidate: str) -> Dict:
        """
        Calculate BERTScore F1 between reference and candidate (Zhang et al. 2019)
        
        Args:
            reference: Ground truth/reference text
            candidate: Generated/candidate text to compare
            
        Returns:
            Dict with bertscore_precision, bertscore_recall, bertscore_f1
        """
        try:
            if not reference or not candidate:
                return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
            
            # Using the official bert_score library (Zhang et al. 2019 implementation)
            P, R, F1 = bert_score([candidate.strip()], [reference.strip()], 
                                 lang='en', 
                                 model_type='bert-base-uncased',
                                 verbose=False)
            
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"Official BERTScore library failed: {e}")
            logger.info("Falling back to manual implementation...")
            return self._manual_bertscore(reference, candidate)
    
    def _manual_bertscore(self, reference: str, candidate: str) -> Dict:
        """Manual implementation of BERTScore as fallback (Zhang et al. 2019)"""
        try:
            if not self.bert_model or not self.bert_tokenizer:
                return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
            
            # Tokenize and get BERT embeddings
            ref_tokens = self.bert_tokenizer(reference.strip(), return_tensors='pt', 
                                           padding=True, truncation=True, max_length=512)
            cand_tokens = self.bert_tokenizer(candidate.strip(), return_tensors='pt', 
                                            padding=True, truncation=True, max_length=512)
            
            ref_tokens = {k: v.to(self.device) for k, v in ref_tokens.items()}
            cand_tokens = {k: v.to(self.device) for k, v in cand_tokens.items()}
            
            with torch.no_grad():
                ref_outputs = self.bert_model(**ref_tokens)
                cand_outputs = self.bert_model(**cand_tokens)
            
            # Use last hidden states as embeddings
            ref_embeddings = ref_outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
            cand_embeddings = cand_outputs.last_hidden_state.squeeze(0)
            
            # Remove padding tokens from calculation
            ref_mask = ref_tokens['attention_mask'].squeeze(0).bool()
            cand_mask = cand_tokens['attention_mask'].squeeze(0).bool()
            
            ref_embeddings = ref_embeddings[ref_mask]
            cand_embeddings = cand_embeddings[cand_mask]
            
            # Calculate cosine similarity matrix
            similarity_matrix = torch.nn.functional.cosine_similarity(
                ref_embeddings.unsqueeze(1), cand_embeddings.unsqueeze(0), dim=2
            )
            
            # Precision: max similarity for each candidate token
            precision = similarity_matrix.max(dim=0)[0].mean().item()
            
            # Recall: max similarity for each reference token  
            recall = similarity_matrix.max(dim=1)[0].mean().item()
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            return {
                'bertscore_precision': precision,
                'bertscore_recall': recall,
                'bertscore_f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error in manual BERTScore: {e}")
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
    
    def compare_all_questions(self, original_rq: str, improved_rq_suggestion: str, 
                             t5_base_rq: str, t5_small_rq: str, mistral_rq: str = "") -> Dict:
        """
        Compare all question variants using multiple reference strategies
        
        Strategy 1: Use Improved RQ Suggestion as reference (human-annotated ground truth)
        Strategy 2: Use Original RQ as reference (improvement from baseline)
        """
        
        results = {}
        
        # Strategy 1: Improved RQ Suggestion as Reference (Primary Analysis)
        print("  Using Improved RQ Suggestion as reference...")
        if improved_rq_suggestion:
            comparisons_vs_improved = {
                'original_vs_improved': (original_rq, improved_rq_suggestion),
                't5_base_vs_improved': (t5_base_rq, improved_rq_suggestion),
                't5_small_vs_improved': (t5_small_rq, improved_rq_suggestion),
                'mistral_vs_improved': (mistral_rq, improved_rq_suggestion) 
            }
            
            for comp_name, (candidate, reference) in comparisons_vs_improved.items():
                if not candidate or not reference:
                    # Fill with NaN for missing data
                    for metric in ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1', 
                                 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
                        results[f'{comp_name}_{metric}'] = np.nan
                    continue
                
                # Calculate ROUGE-L (Lin 2004)
                rouge_scores = self.calculate_rouge_l(reference, candidate)
                results.update({f'{comp_name}_{k}': v for k, v in rouge_scores.items()})
                
                # Calculate BERTScore F1 (Zhang et al. 2019)
                bert_scores = self.calculate_bertscore_f1(reference, candidate)
                results.update({f'{comp_name}_{k}': v for k, v in bert_scores.items()})
        
        # Strategy 2: Original RQ as Reference (Secondary Analysis)
        print("  Using Original RQ as reference...")
        if original_rq:
            comparisons_vs_original = {
                'improved_vs_original': (improved_rq_suggestion, original_rq),
                't5_base_vs_original': (t5_base_rq, original_rq),
                't5_small_vs_original': (t5_small_rq, original_rq),
                'mistral_vs_original': (mistral_rq, original_rq) 
            }
            
            for comp_name, (candidate, reference) in comparisons_vs_original.items():
                if not candidate or not reference:
                    # Fill with NaN for missing data
                    for metric in ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1',
                                 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
                        results[f'{comp_name}_{metric}'] = np.nan
                    continue
                
                # Calculate ROUGE-L (Lin 2004)
                rouge_scores = self.calculate_rouge_l(reference, candidate)
                results.update({f'{comp_name}_{k}': v for k, v in rouge_scores.items()})
                
                # Calculate BERTScore F1 (Zhang et al. 2019)
                bert_scores = self.calculate_bertscore_f1(reference, candidate)
                results.update({f'{comp_name}_{k}': v for k, v in bert_scores.items()})
        
        
        # --- Strategy 3: Cross-model comparisons ---
        # Additional: Direct comparison between generated questions
        print("  Comparing T5 models...")
        if t5_base_rq and t5_small_rq:
            # T5-Base as reference, T5-Small as candidate
            rouge_scores = self.calculate_rouge_l(t5_base_rq, t5_small_rq)
            results.update({f't5_small_vs_t5_base_{k}': v for k, v in rouge_scores.items()})
            
            bert_scores = self.calculate_bertscore_f1(t5_base_rq, t5_small_rq)
            results.update({f't5_small_vs_t5_base_{k}': v for k, v in bert_scores.items()})
        
        if mistral_rq and t5_small_rq:   
            rouge = self.calculate_rouge_l(mistral_rq, t5_small_rq)
            bert = self.calculate_bertscore_f1(mistral_rq, t5_small_rq)
            results.update({f'mistral_vs_t5_small_{k}': v for k, v in rouge.items()})
            results.update({f'mistral_vs_t5_small_{k}': v for k, v in bert.items()})

        if mistral_rq and t5_base_rq:    
            rouge = self.calculate_rouge_l(mistral_rq, t5_base_rq)
            bert = self.calculate_bertscore_f1(mistral_rq, t5_base_rq)
            results.update({f'mistral_vs_t5_base_{k}': v for k, v in rouge.items()})
            results.update({f'mistral_vs_t5_base_{k}': v for k, v in bert.items()})
        
        return results
    
    def process_main_output(self, main_output_path: str,mistral_output_path: str, output_path: str):
        """
        Process the output Excel file from main.py and calculate all comparisons
        
        Args:
            main_output_path: Path to Excel file from main.py (contains all RQ variants)
            output_path: Path where comparison results will be saved
        """
        #load for main T5 small and base
        print(f"Loading data from main.py output: {main_output_path}")
        try:
            df = pd.read_excel(main_output_path)
            print(f"Loaded {len(df)} examples")
            
            # Show available columns
            print(f"Available columns: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading main output data: {e}")
            return None
        
        #load for mistral
        print(f"Loading Mistral results: {mistral_output_path}")
        try:
            df_mistral = pd.read_excel(mistral_output_path)
            print(f"Loaded {len(df_mistral)} examples from Mistral output")
            print(f"Available columns in Mistral: {list(df_mistral.columns)}")
        except Exception as e:
            logger.error(f"Error loading Mistral output data: {e}")
            df_mistral = None  # Will skip Mistral comparisons if not loaded
        
        results = []
        
        print(f"\nProcessing {len(df)} research questions...")
        print("="*60)
        
        for idx, row in df.iterrows():
            if idx % 5 == 0:  # Progress indicator
                print(f"Processing example {idx + 1}/{len(df)}")
            
            # Extract all RQ variants from main.py output
            original_rq = str(row['Original_Question']) if 'Original_Question' in row and pd.notna(row['Original_Question']) else ""
            improved_rq = str(row['Improved_RQ_Suggestion']) if 'Improved_RQ_Suggestion' in row and pd.notna(row['Improved_RQ_Suggestion']) else ""
            t5_base_rq = str(row['T5_Base_Improved']) if 'T5_Base_Improved' in row and pd.notna(row['T5_Base_Improved']) else ""
            t5_small_rq = str(row['T5_Small_Improved']) if 'T5_Small_Improved' in row and pd.notna(row['T5_Small_Improved']) else ""
            
            # Extract Mistral generated RQ (assuming row order matches)
            mistral_rq = ""
            if df_mistral is not None and 'Generated_RQ' in df_mistral.columns:
                mistral_row = df_mistral.iloc[idx]
                mistral_rq = str(mistral_row['Generated_RQ']) if pd.notna(mistral_row['Generated_RQ']) else ""
            
            # Skip if no questions to compare
            if not any([original_rq, improved_rq, t5_base_rq, t5_small_rq]):
                print(f"  Skipping row {idx} - no questions available")
                continue
            
            print(f"  Row {idx}: Calculating ROUGE-L and BERTScore...")
            
            # Calculate all comparisons
            comparison_results = self.compare_all_questions(
                original_rq, improved_rq, t5_base_rq, t5_small_rq, mistral_rq
            )
            
            # Prepare result row
            result_row = {
                'row_index': idx,
                'original_rq': original_rq,
                'improved_rq_suggestion': improved_rq,
                't5_base_generated': t5_base_rq,
                't5_small_generated': t5_small_rq,
                'mistral_generated': mistral_rq,
                'title': row.get('Title', ''),
                'link': row.get('Link', '')
            }
            
            # Add comparison metrics
            result_row.update(comparison_results)
            results.append(result_row)
        
        if not results:
            print("No valid comparisons found!")
            return None
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        print("\nCalculating summary statistics...")
        summary_stats = self._calculate_summary_statistics(results_df)
        
        # Add summary row
        results_df = pd.concat([results_df, pd.DataFrame([summary_stats])], ignore_index=True)
        
        # Save results
        results_df.to_excel(output_path, index=False)
        print(f"✓ Comparison results saved to: {output_path}")
        
        # Print detailed summary
        self.print_detailed_summary(results_df)
        
        return results_df
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate mean, std, and other summary statistics"""
        
        summary = {'row_index': 'SUMMARY_STATS'}
        
        # Get all metric columns
        metric_columns = [col for col in df.columns if any(x in col for x in ['rouge_l', 'bertscore'])]
        
        for col in metric_columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                summary[f'{col}'] = valid_values.mean()
                summary[f'{col}_std'] = valid_values.std()
                summary[f'{col}_count'] = len(valid_values)
        
        return summary
    
    def print_detailed_summary(self, results_df: pd.DataFrame):
        """Print comprehensive summary of comparison results"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON RESULTS SUMMARY")
        print("="*80)
        
        # Get summary row (last row)
        summary_row = results_df.iloc[-1]
        
        print("\n PRIMARY ANALYSIS: Using Improved RQ Suggestion as Reference")
        print("-" * 60)
        
        primary_comparisons = ['original_vs_improved', 't5_base_vs_improved', 't5_small_vs_improved', 'mistral_vs_improved']
        
        for comp in primary_comparisons:
            rouge_f1 = summary_row.get(f'{comp}_rouge_l_f1', np.nan)
            bert_f1 = summary_row.get(f'{comp}_bertscore_f1', np.nan)
            count = summary_row.get(f'{comp}_rouge_l_f1_count', 0)
            
            print(f"\n{comp.replace('_', ' ').title()}:")
            print(f"  ROUGE-L F1: {rouge_f1:.3f} (n={count})")
            print(f"  BERTScore F1: {bert_f1:.3f}")
        
        print("\n SECONDARY ANALYSIS: Using Original RQ as Reference")
        print("-" * 60)
        
        secondary_comparisons = ['improved_vs_original', 't5_base_vs_original', 't5_small_vs_original', 'mistral_vs_original']
        
        for comp in secondary_comparisons:
            rouge_f1 = summary_row.get(f'{comp}_rouge_l_f1', np.nan)
            bert_f1 = summary_row.get(f'{comp}_bertscore_f1', np.nan)
            count = summary_row.get(f'{comp}_rouge_l_f1_count', 0)
            
            print(f"\n{comp.replace('_', ' ').title()}:")
            print(f"  ROUGE-L F1: {rouge_f1:.3f} (n={count})")
            print(f"  BERTScore F1: {bert_f1:.3f}")
        
        print("\n MODEL COMPARISON: T5-Base vs T5-Small")
        print("-" * 60)
        
        t5_rouge = summary_row.get('t5_small_vs_t5_base_rouge_l_f1', np.nan)
        t5_bert = summary_row.get('t5_small_vs_t5_base_bertscore_f1', np.nan)
        print(f"T5-Small vs T5-Base:")
        print(f"  ROUGE-L F1: {t5_rouge:.3f}")
        print(f"  BERTScore F1: {t5_bert:.3f}")
        
        # Model comparisons involving Mistral
        print("\n MODEL COMPARISONS: Mistral vs T5 Models")
        print("-" * 60)
        
        mistral_t5_small_rouge = summary_row.get('mistral_vs_t5_small_rouge_l_f1', np.nan)
        mistral_t5_small_bert = summary_row.get('mistral_vs_t5_small_bertscore_f1', np.nan)
        print(f"Mistral vs T5-Small:")
        print(f"  ROUGE-L F1: {mistral_t5_small_rouge:.3f}")
        print(f"  BERTScore F1: {mistral_t5_small_bert:.3f}")
        
        mistral_t5_base_rouge = summary_row.get('mistral_vs_t5_base_rouge_l_f1', np.nan)
        mistral_t5_base_bert = summary_row.get('mistral_vs_t5_base_bertscore_f1', np.nan)
        print(f"Mistral vs T5-Base:")
        print(f"  ROUGE-L F1: {mistral_t5_base_rouge:.3f}")
        print(f"  BERTScore F1: {mistral_t5_base_bert:.3f}")
        
        print("\n BEST PERFORMING MODEL")
        print("-" * 60)
        
        # Compare T5 models against improved reference
        t5_base_bert = summary_row.get('t5_base_vs_improved_bertscore_f1', 0)
        t5_small_bert = summary_row.get('t5_small_vs_improved_bertscore_f1', 0)
        mistral_bert = summary_row.get('mistral_vs_improved_bertscore_f1', 0)
        
        models = {
            'T5-Base': t5_base_bert,
            'T5-Small': t5_small_bert,
            'Mistral': mistral_bert
        }
        
        best_model = max(models, key=models.get)
        best_score = models[best_model]
        
        if not np.isnan(best_score):
            print(f" Best performing model: {best_model} (BERTScore F1: {best_score:.3f})")
            for model, score in models.items():
                if model != best_model:
                    print(f"  {model}: {score:.3f}")
        else:
            print(" No valid scores for model comparison.")
        
        print("\n" + "="*80)

def main():
    """Main function to run comparison on main.py output"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize comparator
    comparator = RQComparator()
    
    # Process main.py output
    main_output_path = 'results/test_analysis.xlsx'  # Output from main.py
    mistral_output_path = 'models/mistral_model/Mistral_Test_Results.xlsx'
    output_path = 'results/rq_comparison_detailed.xlsx'
    
    if not os.path.exists(main_output_path):
        print(f" Main output file not found: {main_output_path}")
        print("Please run main.py first to generate the test analysis.")
        return
    
    if not os.path.exists(mistral_output_path):
        print(f" Warning: Mistral output file not found: {mistral_output_path}")
        print(" Mistral comparisons will be skipped. Please run mistral.py first.")
    
    print("Starting comprehensive RQ comparison analysis...")
    print("- Implementing ROUGE-L (Lin 2004)")
    print("- Implementing BERTScore F1 (Zhang et al. 2019)")
    print("- Multiple reference strategies")
    print()
    
    # Process the data
    results_df = comparator.process_main_output(main_output_path, mistral_output_path, output_path)
    
    if results_df is not None:
        print(f"\n Analysis complete! Results saved to: {output_path}")
        print("Summary statistics included in the output file.")
    else:
        print("Analysis failed. Please check the input file format.")

if __name__ == "__main__":
    main()