## float_point_comparison / recall_results

## Objective

Quick preview experiment to compare generation outputs and evaluation coverage across three configurations:
- `hf` — HuggingFace pipeline run
- `vllm` — vLLM-backed run
- `vllm_float16` — vLLM run using float16 precision

The notebook `performance_evaluation.ipynb` computes aggregated metrics (the notebook uses `rouge_l` via
`utils.metrics.calculate_metric.calculate_agg_metric`) and writes per-experiment files named `result_<exp>.feather`.
This README focuses on the number of valid (non-null / non-zero) metric values produced per experiment, which indicates
how many completed model outputs are available for fair comparison.

## Subfolder structure

- `generations/` — raw model output files (JSON/NDJSON) produced by each run
- `results/` — where notebook output files may be stored (e.g., `result_hf.feather`, `result_vllm.feather`, ...)
- `logs/` — run logs and any WandB offline artifacts
- `performance_evaluation.ipynb` — notebook used to compute metrics and produce quick summaries

## Why counts are the primary focus here

- We care about the number of valid metric rows per experiment because it reflects how many examples produced usable outputs.
  A higher count gives more confidence in comparisons; a small count means the experiment produced few usable outputs and
  should be treated cautiously.

## How the notebook computes counts (Polars examples)

- To create `result_<exp>.feather` files the notebook runs `calculate_agg_metric` over the `generations/` files. After loading
  the per-experiment feather files into a `results` DataFrame, use these snippets to count valid (non-zero) metric values:

  results.filter(pl.col("mean") > 0).group_by("exp_name").agg(pl.col("mean").count())

## Results

Main results:
- The `hf` experiment produced considerably more than the default vLLM and the vLLM float16 runs.

## Notes

- The notebook currently uses `rouge_l` as the metric passed to `calculate_agg_metric` but this README intentionally
  focuses on counting non-null results instead of metric central tendency.
- Check `performance_evaluation.ipynb` for the exact code used to generate `result_<exp>.feather` files if you need to re-run
  metric generation.
