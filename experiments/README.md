# Experiments

This folder aggregates all experiments running with the Datamodels for Context Reduction (dmcr) using the Natural Questions dataset.

## prompt_validation_experiment

**Objective**:

**Setup**:

**Folder organization**:

**Results**:

## selected_question_experiment

**Objective**:

**Setup**:

**Folder organization**:

**Results**:

## subpartition_rag_datamodel

**Objective**:

**Setup**:

**Folder organization**:

**Results**:

## rag_prompts_comparison

**Objective**: This experiment was originated from the "subpartition_rag_datamodel". We want to compare our proposed instruction with the one presented in the paper [Power of Noise](https://dl.acm.org/doi/10.1145/3626772.3657834) in order to have a direct comparison of metric for current used model

**Setup**: The used model was a  Llama-3-8B-Intruct with quantization 8-bit, th

**Folder organization**:

**Results**:

## poc_perplexity_proxy_groundtruth

**Creation Date**: 30-06-2025

**Objective**: This experiment aims to evaluate the use of perplexity as a proxy for retrieval-augmented generation (RAG) model performance. The core idea is to assess whether lower perplexity scores on a language model, given a set of retrieved documents, correlate with higher-quality generated answers. For more details, see the [experiment's README](./poc_perplexity_proxy_groundtruth/README.md).

**Setup**: The experiment uses the `Llama-3.2-3B-Instruct` model and its tokenizer. The main libraries used are `accelerate` for distributed training, `polars` for data manipulation, `numpy` for numerical operations, `torch` for the deep learning framework, `h5py` for handling HDF5 files, and `tyro` for command-line argument parsing.

**Folder organization**: The experiment is organized into the following directories:
- **`experiments_.../`**: Each experiment has its own directory, identified by a unique seed.
    - **`datamodels/`**: Contains the data models and collections for the experiment.
        - **`collections/`**: Contains the generated collections of documents for each experiment.
        - **`models/`**: Contains the trained models.
    - **`generations/`**: Contains the generated answers for each run type.
    - **`retrieval/`**: Contains the retrieval indexes used by the RAG models.
- **`results/`**: Stores the evaluation results of the experiments in Feather format.

**Results**: The `datamodels` run consistently achieves the highest ROUGE-L scores across all seeds, indicating that the data models significantly improve the quality of the generated answers. The `perplexity` runs, on the other hand, perform poorly, which is expected since it is not a generation model but a metric. The `rag` and `baseline` runs show similar performance, with the `rag` model having a slight edge. For a detailed table with the results, please refer to the [experiment's README](./poc_perplexity_proxy_groundtruth/README.md#rouge-l-scores).


## qualitative_analysis_context_quality

**Creation Data**: 22-08-2025

**Objective**: We have strong indications in small and controlled sets that points that the datamodels (knowing the groundtruth) can retrieve a certain context and achieve better reuslts than the naive RAG or baseline. The goal of this notebook is to undestand the differences of the context retrievers, how far away is the context retrieved through internal product from the relations encountered by the datamodels

**Folder organization**:
- **`experiments`/**: contains the experiments with the generations and the distances/indexes returned for both RAG and Datamodels
- **global_comparison.ipynb**: 
- **showcases.ipynb**:

**Results**::