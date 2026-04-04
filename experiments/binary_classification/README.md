# Binary Classification Experiment

This experiment evaluates binary classification models for assessing the quality of RAG (Retrieval-Augmented Generation) based responses in the Natural Questions (NQ) Open dataset. The goal is to classify responses as either helpful/correct (1) or unhelpful/incorrect (0).

## Experiment Objective

## Setup

This experiment is based on the the RAG-base retrieval e groundtruth collections from the `analysing_datamodels_training` experiment, which uses ROUGE-L scores as the basis for binary classification. In order to be able to reproduct this experiment is necessary to:
1. Run the `analysing_datamodels_training` experiment to generate the necessary ROUGE-L ground truth collections. Copy the generated `collections/groundtruth/` directory to `rougel_groundtruth/` in this experiment and copy the `runs`into `datamodels_runs/` in this experiment. An important details is that the retrieved documents and RAG generations come from `analysing_datamodels_training`.
2. Create the judge collections from the copied judge structure or run each step for different models (Qwen versions) or instructions (extraction instruction). See `scripts` for utility scripts to create the datamodels setup, pre-collections and collections.
3. Run the `create_binary_collections.py` script to create the binary classification datasets, for the ROUGE-L any metric above 0 is considered a positive example (1) and 0 is considered a negative example (0). 
4. The analysis of collections train/test distribution, ROC-AUC and text-generation results is performed on `notebooks/binary_classification_analysis.ipynb`, for Llama-3.2-3B-Instruct, and in `notebooks/qwen_classification.ipynb`for Qwen3-8B. Additional analysis based on confusion matrices between the different collections is performed in `notebooks/confusion_matrices.ipynb`.

## Repository Structure

```
binary_classification/
├── create_binary_collections.py      # Script to create binary collections from raw data
├── run_datamodels.py                 # Main script to run RAG-based generation pipeline
├── run_logreg_gridsearch.py          # Grid search for logistic regression hyperparameters
├── binary_collections/               # Binary classification datasets
│   ├── groundtruth/                 # Ground truth based collections
│   ├── judge/                       # Judge-based classifications
│   ├── alt1/                        # Alternative prompt format 1 (ALT1)
│   ├── alt2/                        # Alternative prompt format 2 (ALT2)
│   ├── voting_alt1/                 # Voting ensemble with ALT1
│   └── voting_qwen/                 # Voting ensemble with Qwen model
├── best_params/                      # Best hyperparameters from grid search
│   ├── best_params_*.json           # Best params from CV grid search
│   └── holdout_best_params_*.json   # Best params using holdout validation
├── generation_results/               # Generated classification results
├── datamodels_runs/                  # Training outputs and checkpoints
├── ebm/                              # Explainable Boosting Machine (EBM) results
├── notebooks/                        # Analysis and visualization notebooks
├── rougel_groundtruth/               # ROUGE-L ground truth evaluations
├── scripts/                          # Utility shell scripts
└── weights/                          # Model weights
```

## Binary Collections

This experiment compares multiple binary classification approaches, each using different evaluation methods or prompt formulations. Each collection contains train/test splits across 5 random seed experiments (experiment_1, experiment_4, experiment_54, experiment_61, experiment_73).

### Collection Types

#### **groundtruth**
- **Description**: Binary classifications derived directly from ROUGE-L ground truth scores
- **Evaluation Method**: ROUGE-L metric on validation set
- **Conversion**: Scores converted to binary (1 if score > 0, else 0)
- **Related Files**: 
  - Created from: `rougel_groundtruth/` directory
  - Best params: `best_params/best_params_groundtruth_exp*.json`
  - Holdout params: `best_params/holdout_best_params_groundtruth_exp*.json`

#### **judge**
- **Description**: Binary classifications using the BinaryJudge evaluator
- **Evaluation Method**: LLM-based judge model (specified via `evaluator: "BinaryJudge"` in config)
- **Prompt Format**: Default judge format that classifies responses as "RESPONDS QUESTION" (1) or "NOT RESPONDS QUESTION" (0)
- **Related Model**: Uses model specified in `language_model_path` (default: Llama-3.2-3B-Instruct)
- **Related Files**:
  - Best params: `best_params/best_params_judge_exp*.json`
  - Generated results: `generation_results/logistic_regression.feather`

#### **alt1** (Alternative Format 1)
- **Description**: Alternative prompt formulation emphasizing strict judgment criteria
- **Evaluation Method**: LLM-based judge with reformatted prompt
- **Prompt Focus**: Impartial evaluation with strict binary classification (RESPONDS vs NOT RESPONDS)
- **Format Function**: `alt_format_input_1()` in `run_datamodels.py`
- **Key Instruction**: "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
- **Related Configuration**: `format_input: "ALT1"` in DatamodelsConfig
- **Use Case**: Testing impact of more formal/structured judge prompts

#### **alt2** (Alternative Format 2)
- **Description**: Alternative prompt formulation emphasizing "Helpfulness-First" metrics
- **Evaluation Method**: LLM-based judge with lenient evaluation criteria
- **Prompt Focus**: Evaluates if response successfully helps user, tolerating minor stylistic issues
- **Format Function**: `alt_format_input_2()` in `run_datamodels.py`
- **Key Principle**: GOOD (1) classification if core question answered correctly; BAD (0) if factually wrong or unhelpful
- **Related Configuration**: `format_input: "ALT2"` in DatamodelsConfig
- **Use Case**: Testing impact of more flexible/pragmatic judge criteria

#### **voting_alt1**
- **Description**: Ensemble voting method combining predictions from ALT1 format models
- **Evaluation Method**: Multiple model predictions with ensemble voting strategy
- **Related Configuration**: Uses multiple checkpoints from ALT1 generation
- **Implementation**: Located in `datamodels_runs/runs/*/datamodels/collections/`
- **Collection Pattern**: Aggregates files matching `"Voting"` pattern with ALT1 format
- **Use Case**: Testing robustness through ensemble voting

#### **voting_qwen**
- **Description**: Ensemble voting using Qwen model-based classifications
- **Evaluation Method**: Qwen model with voting aggregation
- **Related Model**: Qwen model (path specific configuration in run_datamodels.py)
- **Use Case**: Cross-model validation and ensemble robustness

### Collection Creation Pipeline

Binary collections are created through the following pipeline:

1. **Input**: Raw evaluation results from RAG pipeline runs
2. **Processing**: 
   - Read feather files from `datamodels_runs/runs/*/datamodels/collections/{train,test}/`
   - Filter by pattern (e.g., "BinaryJudge_", "ALT1", "ALT2", "Voting")
   - Unify checkpoints per split (concatenate multiple model checkpoints)
3. **Binary Conversion**: Convert continuous evaluation scores to binary (0 or 1)
4. **Output**: Saved to `binary_collections/{type}/experiment_*/train.feather` and `test.feather`

### Experiments Configuration

Each binary collection is tested across 5 experiments with different random seeds:
- **experiment_1** (seed=1)
- **experiment_4** (seed=4)
- **experiment_54** (seed=54)
- **experiment_61** (seed=61)
- **experiment_73** (seed=73)

This provides robustness across different random initializations.

### Hyperparameter Tuning

For each binary collection type and experiment, logistic regression hyperparameters are optimized using grid search with Stratified K-Fold cross-validation:

- **Parameter Grid**:
  - `solver`: [lbfgs, newton-cg, newton-cholesky, sag, liblinear, saga]
  - `C` (inverse regularization strength): $10^{-3}$ to $10^{3}$ (7 values)
  - `l1_ratio` (elasticnet penalty mix): [0] (L2), [0, 1] (liblinear), or [0, 0.2, 0.5, 0.8, 1] (saga)

- **Results**: Best hyperparameters stored in `best_params/best_params_{collection_type}_exp{i}.json`

## Key Configuration Parameters

The experiment is configured in `DatamodelsConfig` dataclass in `run_datamodels.py`:

- `run_type`: Type of pipeline stage (setup, pre_collections, collections, training, generation)
- `format_input`: Prompt format (None, "ALT1", "ALT2")
- `model_tag`: Model identifier
- `seed`: Random seed for reproducibility
- `evaluator`: Evaluation method (e.g., "BinaryJudge")
- `language_model_path`: Path to LLM (default: models/Llama-3.2-3B-Instruct)
- `embedder_path`: Path to embedding model (default: models/bge-base-en-v1.5)
- `k`: Number of retrieved passages (default: 16)

## Running the Experiment

### Generate Binary Collections
```bash
python create_binary_collections.py
```

### Run Datamodels Pipeline
```bash
python run_datamodels.py --run_type collections --evacuator BinaryJudge
```

### Grid Search for Best Hyperparameters
```bash
python run_logreg_gridsearch.py --subfolder judge --exp-index 0
```

## Output Files

- **Binary Collections**: `binary_collections/{type}/experiment_{i}/{train,test}.feather`
- **Best Parameters**: `best_params/best_params_{type}_exp{i}.json`
- **Generation Results**: `generation_results/*.feather`
- **Model Weights**: `weights/{model_name}.pt`
- **Notebooks**: `notebooks/*.ipynb` (analysis and visualization)

## Related Experiments

- **Ablation Performance**: `../ablation_performance/` - Systematic evaluation of different model combinations
