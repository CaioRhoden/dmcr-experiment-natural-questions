## multiple_sentence_inference

## Objective

Brief experiment to debug and compare RAG (retrieval-augmented generation) recall and generation between vllm-based pipelines and Hugging Face (HF) pipelines when performing multiple-sentence inference. The folder contains runs, logs, retrieval outputs, and helper scripts used to reproduce and inspect differences across models/configurations.

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


What each run means (inferred from filenames and folder structure):

- `vllm3` / `vllm5`: runs using the vllm-backed pipeline. The numeric suffix likely denotes the number of generated sentences or the inference configuration (3 vs 5) used for multi-sentence inference.
- `batch-hf3` / `batch-hf5`: Hugging Face pipeline runs executed in batched mode; suffix again indicates the 3 vs 5 configuration.
- `single-hf`: Hugging Face pipeline run executed in single-instance (non-batched) mode. 


Main conclusions:

- The increase number of generated sentences presented a considerable improvement in the returno of non-null Rouge-L scores across all configurations.
- The difference between the numer of inferences for the HF pipelines is not that impactful, but it's super relevant for the vllm


## Next steps:

- If you want the README to include the actual numeric values, either open the `result_*.feather` files and paste the `mean` numbers here, or run the notebook/code that the repo already contains to re-generate the `result_*.feather` files and then re-run the aggregation cell.
- The vLLM pipeline seems to be a better option due to the inference speedup it provides when scaling to multiple sentences.
