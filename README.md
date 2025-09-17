# Research Question Scoring and Improvement

This project explores automatic scoring and rewriting of research questions using large language models (LLMs).
It combines BERT-based classifiers, T5 generators, and a fine-tuned Mistral-7B model.
The workflow is designed to run on a SLURM HPC cluster, but supports interactive/local testing as well.

## Project Structure:

.
├── ScoringModel.py # BERT + T5 scoring models
├── QstImprModel.py # T5-based improvement models
├── main.py # Main entrypoint (scoring + rewriting pipeline)
├── compareRQ.py # Evaluation of rewritten vs reference questions
├── statisticalTest.py # Statistical tests on model-generated scores
├── mistral.py # Fine-tuning + testing Mistral-7B
├── training.py # Training pipeline for BERT/T5 scorers
├── \*.slurn # SLURM job scripts for cluster execution

## Workflow Overview

1.  **Train models**
    Fine-tune the scoring models (BERT/T5).
    Run via SLURM: sbatch trainModels.slurn

2.  **Running the main pipeline:**

    Apply scoring + rewriting to test questions.

    Note that main.py has 2 execution modes available:

        a. **Batch mode (test)**
        Runs on a fixed test dataset of research questions.
        Useful for automated experiments and SLURM jobs.

        b. **Interactive mode (interactive)**
        Lets you input your own research questions.
        You’ll get predicted scores + improved versions in real-time.

    Run via SLURM: sbatch main.slurn

    OR

    Can run interactively (on local machine or login node):
    python main.py test # Run on pre-defined test set
    python main.py interactive # Ask your own research questions

3.  **Train and run Mistral**
    Parameter-efficient fine-tuning of Mistral-7B-Instruct (LoRA).
    Run via SLURM: sbatch mistral.slurn
    Produces training history plots, predictions (Excel), and JSON logs.

4.  **Compare re-written questions**
    Evaluate improved RQs vs reference RQs using BERTScore and ROUGE-L.
    run in terminal after step 1-3: python compareRQ.py
    This outputs similarity metrics for each model’s rewrites.

5.  **Comparing Scores/ Statistical Tests**
    Check the similarity of predicted scores vs ground truth.
    Uses paired t-tests and Pearson correlations.
    run in terminal: python statisticalTest.py
    Note: The script includes pre-computed values. If you re-run models and generate new scores, update the lists inside this script.

## Notes on SLURM Usage:

    Most heavy experiments are launched using .slurn scripts (e.g., trainModels.slurn, main.slurn, mistral.slurn).

    Each SLURM script sets up the environment, loads the model, and runs the corresponding Python file.

    For smaller experiments or debugging, you can run the Python scripts directly.

## Authentication:

    For Hugging Face models like Mistral in this case, you’ll need to export your HF token before running:  export HUGGING_FACE_HUB_TOKEN="your_hf_token"

## Additional info

RQ_Feedback.xlsx was added for clarity of the research questions used during this project. It was split into CleanedRQTrain.xlsx for training the models and CleanedRQTest.xlsx for testing the models.

ScoreComparisons.xlxs was added for clarity on where The numbers used in statisticalTest.py were generated from. Only those under average were the ones used for the 4 models.
