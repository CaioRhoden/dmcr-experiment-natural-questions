# Judge Collections Pipeline

This pipeline evaluates predictions from pre-collections using various judge types and creates evaluation results.

## Overview

The `run_judge_collections.py` script provides a complete pipeline to:

1. Load pre-collections data (with `predicted_output` column as `list[str]`)
2. Run judge evaluations on each prediction using a language model
3. Create an `evaluation` column with the judgement scores
4. Explode rows if multiple evaluation values exist per prediction
5. Save the evaluated dataset as a Feather file

## Configuration

The pipeline is configured using `JudgeCollectionsConfig` dataclass with the following options:

### Core Parameters

- **judge_type** (str): Type of judge to use
  - `"PromptJudge"`: Evaluates predictions based on question and prediction only
  - `"PairwiseJudge"`: Compares two predictions (prediction vs reference)
  - `"ContextJudge"`: Evaluates predictions in context of provided context

- **prompt_intructions** (str): Which prompt template to use
  - `"naive_judge"`: General quality evaluation
  - `"recall_naive_judge"`: Evaluates information recall
  - `"pairwise_judge"`: For pairwise comparison (used with PairwiseJudge)
  - `"faithfulness_judge"`: Evaluates faithfulness to context

- **saving_path** (str): Where to save the output Feather file
  - Default: `"judge_collections/default.feather"`

### Model Parameters

- **model_path** (str): Path to the language model in the models directory
  - Default: `"Llama-3.2-3B-Instruct"`
  - Examples: `"Qwen3-4B-Instruct-2507"`, `"gpt-oss-20b"`

- **batch_size** (int): Batch size for model inference
  - Default: `16`

- **regex_pattern** (str): Regex pattern to extract scores from model output
  - Default: `r'\[\[(\d+)\]\]'` (extracts numbers from `[[score]]` format)

### Data Parameters

- **pre_collections_path** (str): Path to pre-collections data directory
  - Default: `"experiment_81/pre_collections/train"`

- **pairwise_rag** (bool): For PairwiseJudge, whether to use RAG-based retrieval
  - Default: `False`

## Usage

### Command Line

```bash
python run_judge_collections.py \
  --judge-type PromptJudge \
  --prompt-intructions naive_judge \
  --model-path Llama-3.2-3B-Instruct \
  --saving-path judge_collections/naive_judge_results.feather \
  --pre-collections-path experiment_81/pre_collections/train
```

### With Different Judges

```bash
# Pairwise comparison
python run_judge_collections.py \
  --judge-type PairwiseJudge \
  --prompt-intructions pairwise_judge \
  --saving-path judge_collections/pairwise_results.feather

# Context-based evaluation
python run_judge_collections.py \
  --judge-type ContextJudge \
  --prompt-intructions faithfulness_judge \
  --saving-path judge_collections/context_results.feather
```

## Input Data Format

The pre-collections dataframe should have the following structure:

```
┌────────────────┬──────────┬─────────────────┬─────────────────────────────────┬────────────────────┐
│ collection_idx ┆ test_idx ┆ input           ┆ predicted_output                ┆ true_output        │
│ ---            ┆ ---      ┆ ---             ┆ ---                             ┆ ---                │
│ i64            ┆ i64      ┆ array[i64, 100] ┆ list[str]                       ┆ list[str]          │
├────────────────┼──────────┼─────────────────┼─────────────────────────────────┼────────────────────┤
│ 0              ┆ 0        ┆ [0, 0, … 0]     ┆ ["Prediction 1"]                ┆ ["True answer"]    │
│ 0              ┆ 1        ┆ [0, 0, … 0]     ┆ ["Pred 1", "Pred 2"]            ┆ ["True answer"]    │
│ 0              ┆ 2        ┆ [0, 0, … 0]     ┆ ["Prediction"]                  ┆ ["True answer"]    │
└────────────────┴──────────┴─────────────────┴─────────────────────────────────┴────────────────────┘
```

- `collection_idx`: Index of the collection/model set
- `test_idx`: Index of the test question
- `input`: Features/embeddings (array of integers)
- `predicted_output`: **List of predictions as strings** (can have multiple predictions per row)
- `true_output`: List of ground truth answers

## Output Data Format

The output dataframe will expand rows based on the number of predictions and evaluation scores:

```
┌────────────────┬──────────┬─────────────────┬──────────────────┬────────────────────┬──────────┬────────────┐
│ collection_idx ┆ test_idx ┆ input           ┆ predicted_output ┆ true_output        ┆ pred_idx ┆ evaluation │
│ ---            ┆ ---      ┆ ---             ┆ ---              ┆ ---                ┆ ---      ┆ ---       │
│ i64            ┆ i64      ┆ array[i64, 100] ┆ str              ┆ list[str]          ┆ i64      ┆ f64       │
├────────────────┼──────────┼─────────────────┼──────────────────┼────────────────────┼──────────┼────────────┤
│ 0              ┆ 0        ┆ [0, 0, … 0]     ┆ "Prediction 1"   ┆ ["True answer"]    ┆ 0        ┆ 0.8       │
│ 0              ┆ 1        ┆ [0, 0, … 0]     ┆ "Pred 1"         ┆ ["True answer"]    ┆ 0        ┆ 0.7       │
│ 0              ┆ 1        ┆ [0, 0, … 0]     ┆ "Pred 2"         ┆ ["True answer"]    ┆ 1        ┆ 0.6       │
│ 0              ┆ 2        ┆ [0, 0, … 0]     ┆ "Prediction"     ┆ ["True answer"]    ┆ 0        ┆ 0.9       │
└────────────────┴──────────┴─────────────────┴──────────────────┴────────────────────┴──────────┴────────────┘
```

- `predicted_output`: **Single prediction string** (expanded from list)
- `pred_idx`: Index of the prediction within the original list
- `evaluation`: Judge's evaluation score (float between 0-1 or integer)

## Key Features

1. **Flexible Judge Types**: Supports PromptJudge, PairwiseJudge, and ContextJudge
2. **List Expansion**: Automatically expands `predicted_output` column (list[str] → str) into separate rows
3. **Multiple Scores**: If a judge returns multiple scores per prediction, creates multiple rows
4. **Model Agnostic**: Works with any model in the models directory
5. **Configurable Regex**: Customize how scores are extracted from model output

## Important Notes

### Question Data

The current implementation requires a valid `questions_list` to be loaded. You need to modify the pipeline to load questions based on your data structure. This is typically:

```python
# In run_judge_collections_pipeline function, replace:
questions_list = []  # TODO: Load actual questions

# With your actual loading logic, e.g.:
# questions_df = pl.read_parquet("path/to/questions.parquet")
# questions_list = questions_df['question'].to_list()
```

### Reference/Context Data

For PairwiseJudge and ContextJudge, you also need to provide:
- `references_list`: Ground truth or reference answers for comparison
- `contexts_list`: Context passages for context-based evaluation

### Model Loading

Models are loaded from the `models/` directory in the project root. Ensure your chosen model is available at `{project_root}/models/{model_path}`.

## Prompts

Prompts are loaded from `judge_prompts.json` in the same directory. You can customize these by editing the JSON file or adding new prompt templates.

## Performance Considerations

- Batch size affects memory usage and speed
- Larger batch sizes = faster but higher memory
- Model choice affects accuracy and speed
- GPU memory utilization is set to 0.9 by default

## Troubleshooting

**Error: "Model not found"**
- Check that the model exists in `models/{model_path}`
- Verify the model_path parameter

**Error: "Invalid regex pattern"**
- Check that the regex_pattern correctly extracts scores from model output
- Test with a sample output first

**Error: "Questions list is empty"**
- Load actual questions data by modifying the pipeline function
- See "Question Data" section above

## Example Workflow

```bash
# 1. Run PromptJudge on all collections
python run_judge_collections.py \
  --judge-type PromptJudge \
  --prompt-intructions naive_judge \
  --saving-path results/prompt_judge.feather

# 2. Run PairwiseJudge for cross-collection comparison
python run_judge_collections.py \
  --judge-type PairwiseJudge \
  --prompt-intructions pairwise_judge \
  --saving-path results/pairwise_judge.feather

# 3. Run Context judge with faithfulness check
python run_judge_collections.py \
  --judge-type ContextJudge \
  --prompt-intructions faithfulness_judge \
  --saving-path results/faithfulness_judge.feather
```
