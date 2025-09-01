# Lasso Regression Experiment

## Objective

This repository contains a series of experiments to evaluate and compare the performance of Lasso regression models against a baseline Linear Regressor. The primary objective is to analyze the impact of different Lasso regularization strengths (alpha) on model performance.

## Experiments Setup

The experiments are organized into separate directories, each representing a specific model configuration:

-   **`linear_regressor/`**: Baseline model without regularization.
-   **`lasso_1/`**: Lasso regression with a specific alpha value.
-   **`lasso_2/`**: Lasso regression with a different alpha value.
-   **`lasso_3/`**: Lasso regression with another alpha value.
-   **`lasso_4/`**: Lasso regression with a fourth alpha value.

Each experiment directory contains:

-   **`pipeline.py`**: A script that defines the data processing, model training, and evaluation pipeline.
-   **`datamodels/`**: Directory containing the train and test data collections, as well as the trained model files.
-   **`generations/`**: Directory containing the model's predictions.
-   **`retrieval/`**: Directory containing data related to the retrieval of information for the models.

## Parameters

The following table summarizes the key parameters used in each experiment:

| Experiment         | Model              | Alpha (Regularization Strength) |
| ------------------ | ------------------ | ------------------------------- |
| `linear_regressor` | Linear Regression  | N/A                             |
| `lasso_1`          | Lasso Regression   | 0.1                             |
| `lasso_2`          | Lasso Regression   | 0.01                            |
| `lasso_3`          | Lasso Regression   | 0.001                           |
| `lasso_4`          | Lasso Regression   | 0.0001                          |

## `pipeline.py` Parameters

This table describes the parameters available in the `pipeline.py` script, which can be used to configure the experiments.

| Parameter             | Type         | Description                                                                                                 |
| --------------------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| `seed`                | `int`        | Random seed for reproducibility.                                                                            |
| `step`                | `str`        | The specific pipeline step to execute (e.g., `setup`, `train`, `evaluate`).                                   |
| `retrieval_path`      | `str`        | Path to the retrieval indexes JSON file.                                                                    |
| `wiki_path`           | `str`        | Path to the wiki dataset file.                                                                              |
| `embeder_path`        | `str`        | Path to the embedder model.                                                                                 |
| `vector_db_path`      | `str`        | Path to the vector database.                                                                                |
| `questions_path`      | `str`        | Path to the questions dataset file.                                                                         |
| `language_model_path`  | `str`        | Path to the language model.                                                                                 |
| `project_log`         | `str`        | Project log name for wandb.                                                                                 |
| `model_run_id`        | `str`        | ID of the model run.                                                                                        |
| `train_collection_id` | `str`        | ID of the training collection.                                                                              |
| `test_collection_id`  | `str`        | ID of the testing collection.                                                                               |
| `k`                   | `int`        | Number of top-k results to retrieve.                                                                        |
| `size_index`          | `int`        | Size of the index.                                                                                          |
| `num_models`          | `int`        | Number of models to use.                                                                                    |
| `evaluation_metric`   | `str`        | Evaluation metric to use.                                                                                   |
| `evaluator`           | `str`        | Evaluator to use.                                                                                           |
| `instruction`         | `str`        | Instruction for the pre-collections step.                                                                   |
| `train_samples`       | `int`        | Number of training samples.                                                                                 |
| `test_samples`        | `int`        | Number of testing samples.                                                                                  |
| `tags`                | `list[str]`  | List of tags for the experiment.                                                                            |
| `train_start_idx`     | `int`        | Starting index for the training set.                                                                        |
| `train_end_idx`       | `int`        | Ending index for the training set.                                                                          |
| `test_start_idx`      | `int`        | Starting index for the testing set.                                                                         |
| `test_end_idx`        | `int`        | Ending index for the testing set.                                                                           |
| `train_checkpoint`    | `int`        | Checkpoint interval for training.                                                                           |
| `test_checkpoint`     | `int`        | Checkpoint interval for testing.                                                                            |
| `lambda_l1`           | `float`      | L1 regularization coefficient for the model.                                                                |
| `epochs`              | `int`        | Number of epochs to train.                                                                                  |
| `lr`                  | `float`      | Learning rate for training.                                                                                 |
| `train_batches`       | `int`        | Number of batches for training.                                                                             |
| `val_batches`         | `int`        | Number of batches for validation.                                                                           |
| `val_size`            | `float`      | Proportion of data for validation.                                                                          |
| `patience`            | `int`        | Patience for early stopping.                                                                                |
| `log_epochs`          | `int`        | Interval for logging.                                                                                       |