# Ablation Performance: Training Size Analysis

## Description

This experiment investigates how the **number of training collections** affects the performance of data models in a Retrieval-Augmented Generation (RAG) system. The primary goal is to understand the scaling behavior of model performance as training data size increases.

### Experimental Setup

- **Training Sizes Tested**: 500, 1K, 2K, 5K, 10K, 15K, and 20K collections
- **Models Per Collection**: 500 logistic regression models (one per collection instance)
- **Evaluators**: Two evaluation methods
  - **Judge**: Binary classification labels from an LLM-based judge (voting_alt1)
  - **Groundtruth**: Binary labels based on ROUGE-L scores (groundtruth)
- **Base Model**: Logistic Regression with hyperparameters optimized via grid search
- **Dataset**: Natural Questions (NQ) Open dataset with Wikipedia retrieval

### Key Components

1. **Binary Collections**: Pre-processed training collections converted to binary classification tasks (label = 1 if evaluation score > 0, else 0)
2. **Datamodels Training**: Multiple logistic regression models trained on different subsets of features/instances
3. **Performance Evaluation**: Models evaluated using ROC-AUC scores on holdout test sets

## Results

### Performance Metrics Summary

The experiment evaluates model performance across different training collection sizes using three evaluation metrics:

#### Datamodel AUC Scores (Mean by Training Size)

| Training Size | Judge AUC | Groundtruth AUC | Delta |
|---------------|-----------|-----------------|-------|
| 500           | 0.56      | 0.77            | +0.21 |
| 1K            | 0.58      | 0.81            | +0.23 |
| 2K            | 0.60      | 0.83            | +0.23 |
| 3K            | 0.61      | 0.83            | +0.22 |
| 5K            | 0.61      | 0.84            | +0.23 |
| 10K           | 0.62      | 0.85            | +0.23 |
| 15K           | 0.63      | 0.85            | +0.22 |
| 19K           | 0.63      | 0.85            | +0.22 |

#### RAG Generation Performance (ROUGE-L Scores by Model Type and Size)

| Model Type | 500 | 1K  | 2K  | 3K  | 5K  | 10K | 15K | 19K |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|
| Judge      | 0.412 | 0.426 | 0.437 | 0.442 | 0.448 | 0.454 | 0.457 | 0.458 |
| Groundtruth| 0.448 | 0.461 | 0.471 | 0.475 | 0.480 | 0.485 | 0.487 | 0.488 |

#### RAG Generation Performance (SQuAD v2 Best Exact by Model Type and Size)

| Model Type | 500 | 1K  | 2K  | 3K  | 5K  | 10K | 15K | 19K |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|
| Judge      | 0.389 | 0.405 | 0.418 | 0.424 | 0.431 | 0.438 | 0.442 | 0.444 |
| Groundtruth| 0.426 | 0.441 | 0.453 | 0.459 | 0.466 | 0.473 | 0.477 | 0.479 |

### Key Findings & Conclusions

1. **Evaluation Source Gap**: Groundtruth models (ROUGE-L based labels) consistently outperform Judge-based models by ~0.22-0.23 AUC points across all training sizes. This suggests that ROUGE-L labels are more predictive and aligned with model training objectives than LLM-based judge labels.

2. **Diminishing Returns Beyond 10K Collections**: Performance gains plateau significantly after 10K-15K training collections:
   - Judge AUC: 0.62 (10K) → 0.63 (19K) = **+1% improvement**
   - Groundtruth AUC: 0.85 (10K) → 0.85 (19K) = **0% improvement**
   - This suggests that computational investment in larger training sets yields minimal returns.

3. **Strong Gains in Low-Data Regime**: Models benefit substantially from increasing training sets up to 10K collections:
   - Judge AUC: 0.56 (500) → 0.62 (10K) = **+10.7% relative improvement**
   - Groundtruth AUC: 0.77 (500) → 0.85 (10K) = **+10.4% relative improvement**
   
### Analysis Notebook
- **Notebook** (`notebooks/exploring_training_sizes.ipynb`): 
  - Loads and analyzes model performance across different training sizes
  - Compares Judge vs Groundtruth evaluators
  - Visualizes scaling trends
  - Computes aggregate metrics (mean AUC, ROC curves, etc.)

## Setup

### Prerequisites

- Python 3.8+
- CUDA support (for GPU-accelerated components like vLLM)
- Conda environment (recommended)

### Environment Setup

1. **Create and activate the Conda environment**:
   ```bash
   conda create -n nq python=3.10
   conda activate nq
   ```

2. **Install dependencies**:
   ```bash
   # From the project root
   pip install -r requirements.txt
   ```

3. **Required Models** (should be in `../../models/`):
   - `Llama-3.2-3B-Instruct`: Language model for generation
   - `bge-base-en-v1.5`: Embedding model for retrieval

4. **Required Data** (should be in `../../data/`):
   - `wiki_dump2018_nq_open/processed/wiki.feather`: Wikipedia dataset
   - `wiki_dump2018_nq_open/processed/wiki_cosine.index`: Vector database

### Directory Structure

```
ablation_performance/
├── README.md                          # This file
├── run_datamodels.py                  # Main experiment runner
├── create_binary_collections.py       # Preprocessing script
├── scripts/
│   ├── generation.sh                  # SLURM script for generation phase
│   ├── pre_collections.sh             # SLURM script for pre-collections phase
│   ├── grid_search.sh                 # Grid search for hyperparameters
│   ├── judge_collections.sh           # Collections creation with judge labels
│   └── rouge_collections.sh           # Collections creation with ROUGE labels
├── best_params/                       # Hyperparameter grid search results
├── binary_collections/                # Preprocessed training data
├── notebooks/
│   └── exploring_training_sizes.ipynb # Analysis and visualization notebook
├── runs/                              # Experiment runs and checkpoints
├── weights/                           # Model weights and biases
└── scripts/                           # Utility scripts
```

## How to Run It

### Phase 1: Setup
Initialize the experiment structure and create data splits:
```bash
python run_datamodels.py \
    --seed 1 \
    --run_type setup
```

### Phase 2: Pre-Collections
Generate collection features/embeddings for train/test splits:
```bash
python run_datamodels.py \
    --seed 1 \
    --run_type pre_collections \
    --start_idx 0 \
    --end_idx 1000 \
    --checkpoint 100 \
    --mode train
```

### Phase 3: Collections
Create binary classification collections from retrieval results:
```bash
python run_datamodels.py \
    --seed 1 \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000 \
    --checkpoint 100 \
    --mode train
```

### Phase 4: Training
Train logistic regression models on collections:
```bash
python run_datamodels.py \
    --seed 1 \
    --run_type training \
    --collection_id "experiment-1_evaluator-Judge"
```

### Phase 5: Generation
Generate predictions using trained models:
```bash
python run_datamodels.py \
    --seed 1 \
    --run_type generation \
    --model_tag "judge_500"
```

### Using SLURM Scripts (Recommended)

For large-scale experiments, use the provided SLURM scripts:

**Pre-collections phase**:
```bash
sbatch scripts/pre_collections.sh
```

**Collections creation**:
```bash
sbatch scripts/judge_collections.sh   # For judge-based labels
sbatch scripts/rouge_collections.sh   # For groundtruth labels
```

**Generation phase**:
```bash
sbatch scripts/generation.sh
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 1 | Random seed (1, 4, 54, 61, 73) |
| `--run_type` | setup | Phase to execute |
| `--k` | 16 | Number of top-k results to retrieve |
| `--num_models` | 500 | Number of models per collection |
| `--train_samples` | 20000 | Total training samples |
| `--test_samples` | 1000 | Total test samples |
| `--batch_size` | 500 | Training batch size |
| `--epochs` | 1000 | Max training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--patience` | 30 | Early stopping patience |
| `--evaluator` | Judge | Evaluator type (Judge or GroundTruth) |

### Analysis with Jupyter Notebook

After training and generation, analyze results in the notebook:

```bash
jupyter notebook notebooks/exploring_training_sizes.ipynb
```

The notebook performs:
- Model loading and evaluation
- AUC score calculation across training sizes
- Performance comparison between Judge and Groundtruth evaluators
- Visualization of scaling trends
- Statistical aggregation of results

### Creating Binary Collections Directly

To preprocess collections without running the full pipeline:

```bash
python create_binary_collections.py
```

This script:
- Unifies checkpoint files per split
- Converts evaluation scores to binary labels
- Saves processed collections to `binary_collections/`

## Expected Workflow

1. **Preprocess data**: Run `setup` to initialize data splits
2. **Generate features**: Run `pre_collections` to create embeddings
3. **Create collections**: Run `collections` to build training data
4. **Train models**: Run `training` to optimize models for each training size
5. **Inference**: Run `generation` to get predictions on test data
6. **Analyze**: Use Jupyter notebook to visualize and compare results

## Notes

- **Memory Requirements**: High-dimensional embeddings and 500 models per size require substantial GPU/CPU memory (~60GB recommended)
- **Computation Time**: Full pipeline can take 48+ hours depending on hardware
- **Random Seeds**: Use provided seeds (1, 4, 54, 61, 73) for reproducibility
- **WANDB Integration**: Logging is set to offline mode (`WANDB_MODE=offline`)
- **Hyperparameters**: Best parameters should be pre-computed via grid search and stored in `best_params/`
