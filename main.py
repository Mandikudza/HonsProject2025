import os
import sys
import pandas as pd
from ScoringModel import ScoringSystem
from QstImprModel import QuestionImprovementSystem, AdvancedImprovementFeatures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ResearchQuestionPipeline:
    """Main pipeline for research question scoring and improvement"""
    
    def __init__(self):
        print("=== Research Question Analysis System ===")
        print("Initializing models...")
        
        # Initialize scoring system
        self.scoring_system = ScoringSystem()
        
        # Initialize improvement system
        self.improvement_system = QuestionImprovementSystem()
        
        print("System ready!\n")
    
    def analyze_single_question(self, question, title="", link="", threshold=4):
        """Analyze a single research question through the complete pipeline"""
        
        logger.info(f"=== ANALYZING RESEARCH QUESTION ===")
        logger.info(f"Question: {question}")
        logger.info(f"Title: {title}")
        logger.info(f"Link: {link}")
        
        # Step 1: Score the question
        logger.info("Step 1: Scoring the question...")
        scores = self.scoring_system.score_question(question, title, link)
        needs_improvement = self.scoring_system.check_needs_improvement(scores, threshold)
        
        # Print score breakdown
        self.scoring_system.print_score_breakdown(question, scores, needs_improvement)
        
        results = {
            'question': question,
            'title': title,
            'link': link,
            'scores': scores,
            'needs_improvement': needs_improvement
        }
        
        # Step 2: Check if improvement is needed
        any_needs_improvement = any(needs_improvement.values())
        
        if any_needs_improvement:
            logger.info(f"\n  Question needs improvement based on scoring results!")
            logger.info("Moving to Question Improvement Model...")
            
            # Step 3: Improve the question
            logger.info("Step 2: Improving the question...")
            
            # Get low scoring areas for targeted improvement
            low_score_areas = self._get_low_score_areas(scores, threshold)
            
            improvement_results = self.improvement_system.improve_with_feedback(
                question, title, low_score_areas
            )
            
            # Print improvement results
            self.improvement_system.print_improvement_results(question, improvement_results)
            
            # Step 4: Analyze improvement quality
            logger.info("Step 3: Analyzing improvement quality...")
            if improvement_results.get('t5_base_improved') or improvement_results.get('t5_small_improved'):
                comparison = AdvancedImprovementFeatures.compare_improvements(
                    improvement_results.get('t5_base_improved'),
                    improvement_results.get('t5_small_improved'),
                    question
                )
                
                logger.info(f"Recommended improvement: {comparison['recommendation']}")
                
                if comparison['recommendation'] == 't5_base':
                    logger.info(f"Final Improved Question: {improvement_results['t5_base_improved']}")
                elif comparison['recommendation'] == 't5_small':
                    logger.info(f"Final Improved Question: {improvement_results['t5_small_improved']}")
            
            results['improvement_results'] = improvement_results
            
        else:
            logger.info(f"\n Question meets all criteria! No improvement needed.")
            results['improvement_results'] = None
        
        return results
    
    def _get_low_score_areas(self, scores, threshold):
        """Identify areas where scores are below threshold"""
        
        low_areas = []
        
        # Check BERT scores
        if scores.get('bert_scores'):
            bert_scores = scores['bert_scores']
            for criterion, score in bert_scores.items():
                if criterion != 'method' and score < threshold:
                    low_areas.append(criterion)
        
        # Check T5 scores (avoid duplicates)
        if scores.get('t5_scores'):
            t5_scores = scores['t5_scores']
            for criterion, score in t5_scores.items():
                if criterion != 'method' and score < threshold and criterion not in low_areas:
                    low_areas.append(criterion)
        
        return low_areas
    
    def analyze_batch_questions(self, questions_data, output_file=None):
        """Analyze multiple questions in batch"""
        
        logger.info(f"=== BATCH ANALYSIS OF {len(questions_data)} QUESTIONS ===")
        
        results = []
        
        for i, data in enumerate(questions_data):
            logger.info(f"\nProcessing question {i+1}/{len(questions_data)}")
            logger.info("-" * 50)
            
            question = data.get('question', '')
            title = data.get('title', '')
            link = data.get('link', '')
            
            result = self.analyze_single_question(question, title, link)
            results.append(result)
            
            logger.info("\n" + "="*50)
        
        # Save results if output file specified
        if output_file:
            self._save_batch_results(results, output_file)
        
        return results
    
    def _save_batch_results(self, results, output_file):
        """Save batch analysis results to Excel"""
        
        # Prepare data for Excel
        excel_data = []
        
        for result in results:
            row = {
                'Original_Question': result['question'],
                'Title': result['title'],
                'Link': result['link']
            }
            
            # Add BERT scores
            if result['scores'].get('bert_scores'):
                bert = result['scores']['bert_scores']
                row.update({
                    'BERT_Relevance': bert.get('relevance'),
                    'BERT_Fluency': bert.get('fluency'),
                    'BERT_Feasibility': bert.get('feasibility'),
                    'BERT_Free_of_Vagueness': bert.get('free_of_vagueness'),
                    'BERT_Needs_Improvement': result['needs_improvement'].get('bert', False)
                })
            
            # Add T5 scores
            if result['scores'].get('t5_scores'):
                t5 = result['scores']['t5_scores']
                row.update({
                    'T5_Relevance': t5.get('relevance'),
                    'T5_Fluency': t5.get('fluency'),
                    'T5_Feasibility': t5.get('feasibility'),
                    'T5_Free_of_Vagueness': t5.get('free_of_vagueness'),
                    'T5_Needs_Improvement': result['needs_improvement'].get('t5', False)
                })
            
            # Add improvement results
            if result.get('improvement_results'):
                imp = result['improvement_results']
                row.update({
                    'T5_Base_Improved': imp.get('t5_base_improved'),
                    'T5_Small_Improved': imp.get('t5_small_improved')
                })
            
            excel_data.append(row)
        
        # Save to Excel
        df = pd.DataFrame(excel_data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_excel(output_file, index=False)
        logger.info(f"Batch results saved to: {output_file}")
    
    def interactive_mode(self):
        """Interactive mode for testing individual questions"""
        
        logger.info("=== INTERACTIVE MODE ===")
        logger.info("Enter research questions for analysis. Type 'quit' to exit.")
        
        while True:
            question = input("Enter research question: ").strip()
            
            if question.lower() == 'quit':
                logger.info("Goodbye!")
                break
            
            if not question:
                logger.info("Please enter a valid question.")
                continue
            
            title = input("Enter paper title : ").strip()
            link = input("Enter paper link (optional): ").strip()
            
            try:
                self.analyze_single_question(question, title, link)
            except Exception as e:
                logger.error(f"Error analyzing question: {e}")
            
            logger.info("\n" + "="*60)

def load_test_data():
    """Load test data from CleanedRQTest.xlsx for demonstration"""
    
    try:
        df = pd.read_excel('CleanedRQTest.xlsx')
        # Map 'y'/'n' to 1/0
        for col in ['Bad RQ', 'Good RQ', 'Perfect RQ']:
            if col in df.columns:
                df[col] = df[col].map({'y': 1, 'n': 0}).fillna(0).astype(int)
        
        test_data = []
        for i in range(len(df)):
            test_data.append({
                'question': df.iloc[i]['RQ'],
                'title': df.iloc[i]['Title'],
                'link': df.iloc[i].get('link', ''),
                'Comments': df.iloc[i].get('Comments', '')
            })
        
        logger.info(f"Loaded {len(test_data)} test questions from CleanedRQTest.xlsx")
        return test_data
    except Exception as e:
        logger.error(f"Could not load test data: {e}")
        return None

def main():
    """Main function"""
    
    print("Research Question Analysis System")
    print("=" * 40)
    
    # Check if models exist
    model_paths = [
        'models/bert_scoring_model',
        'models/t5_scoring_model',
        'models/t5_base_improvement_model',
        'models/t5_small_improvement_model'
    ]
    
    missing_models = [path for path in model_paths if not os.path.exists(path)]
    
    if missing_models:
        logger.error("Some models are missing. Please run training.py first to train the models.")
        logger.error("Missing models:")
        for model in missing_models:
            logger.error(f"  - {model}")
        logger.error("To train models, run: python training.py")
        return
    
    # Initialize pipeline
    try:
        pipeline = ResearchQuestionPipeline()
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        logger.error("Please ensure all models are properly trained.")
        return
    
    # Command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'interactive':
            pipeline.interactive_mode()
        
        elif mode == 'test':
            # Test with sample data
            test_data = load_test_data()
            if test_data:
                pipeline.analyze_batch_questions(test_data, 'results/test_analysis.xlsx')
            else:
                logger.error("No test data available. Using interactive mode.")
                pipeline.interactive_mode()
        
        elif mode == 'batch' and len(sys.argv) > 2:
            # Batch process from file
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'results/batch_analysis.xlsx'
            
            try:
                df = pd.read_excel(input_file)
                # Map 'y'/'n' to 1/0
                for col in ['Bad RQ', 'Good RQ', 'Perfect RQ']:
                    if col in df.columns:
                        df[col] = df[col].map({'y': 1, 'n': 0}).fillna(0).astype(int)
                
                questions_data = []
                for i in range(len(df)):
                    questions_data.append({
                        'question': df.iloc[i]['RQ'],
                        'title': df.iloc[i]['Title'],
                        'link': df.iloc[i].get('link', ''),
                        'Comments': df.iloc[i].get('Comments', '')
                    })
                
                pipeline.analyze_batch_questions(questions_data, output_file)
            except Exception as e:
                logger.error(f"Error processing batch file: {e}")
        
        else:
            logger.error("Usage:")
            logger.error("  python main.py interactive    # Interactive mode")
            logger.error("  python main.py test          # Test with sample data")
            logger.error("  python main.py batch <input_file> [output_file]  # Batch processing")
    
    else:
        # Default: single question demo
        logger.info("\n=== DEMO MODE ===")
        logger.info("Testing with a sample question...")
        
        sample_question = "What are the effects of social media?"
        sample_title = "Social Media Impact on Mental Health"
        
        pipeline.analyze_single_question(sample_question, sample_title)
        
        logger.info("\n" + "="*60)
        logger.info("Demo complete! Use command line arguments for other modes:")
        logger.info("  python main.py interactive")
        logger.info("  python main.py test")

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run main function
    main()