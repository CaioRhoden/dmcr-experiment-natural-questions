## vllm_parameters
## vllm_parameters

## Objective

Brief experiment to debug and compare RAG (retrieval-augmented generation) recall and generation between vllm-based pipelines when using the following different inference parameters:
- "temperature": 0.6, 0.7, 0.8, 0.9, 1.0
- "top_p": 0.7, 0.8, 0.9
- "repetition_penalty": 1.0, 1.05, 1.1, 1.2, 1.5
- "frequency_penalty": 0.0, 0.05, 0.1, 0.2, 0.25
- "presence_penalty": 0.0, 0.05, 0.1, 0.2, 0.25

## Subfolder structure

- `run_rag.py` — main runner script used to launch the RAG experiment in this folder.
- `run.sh` — convenience wrapper to run the experiment.
- `scripts/` — helper shell scripts for different run configurations:
  - `run_rag_hf_single.sh`, `run_rag_hf3.sh`, `run_rag_hf5.sh`, `run_rag_vllm3.sh`, `run_rag_vllm5.sh`

- `vllm3/`, `vllm5/` — outputs for vllm-backed runs
  - `generations/` — model generation outputs (e.g., `rag.json`)
  - `logs/wandb/` — WandB offline run folders and run artifacts
  - `retrieval/` — retrieval diagnostics (e.g., `rag_retrieval_distances.json`, `rag_retrieval_indexes.json`)

- `batch-hf3/`, `batch-hf5/` — batched HF pipeline run outputs
  - `generations/` — generated outputs for batched HF runs
  - `logs/wandb/` — WandB offline run artifacts and logs
  - `retrieval/` — retrieval outputs for the batched HF experiments

## Notes

- Many WandB runs are stored offline under `logs/wandb/` for later inspection. Retrieval JSON files contain distances and indexes helpful for recall analysis.
- This README is intentionally brief — check the scripts and the `retrieval/` and `generations/` folders for concrete outputs and run metadata.

## Results

The notebook `results/performance_evaluation.ipynb` generates per-experiment files named `result_<exp>.feather` using
`utils.metrics.calculate_metric.calculate_agg_metric`. Those files are concatenated into a `results` table. For this README
we focus only on the number of valid (non-null / non-zero) metric values produced per experiment rather than any mean values.

Main results:
- The "top_p" variation produced the highest impact on generation recall, with lower top_p values (0.7, 0.8) producing more valid generations than higher values (0.9).
- Small variations on "presence_penalty" and "frequency_penalty" also produced some impact on generation recall, with slightly higher values (0.05) producing more valid generations than zero values.

## Next steps
- The parameters variation will be used in a validation datamodels experiment. 
