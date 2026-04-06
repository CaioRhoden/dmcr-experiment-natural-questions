# NQ-Open with Reference

This directory has as the goal compile the results of the DMCR with reference for the [NQ-Open dataset](https://huggingface.co/datasets/google-research-datasets/nq_open). We want to analyze the results of the DMCR re-ranking in relation to the retrieved passages in order to understand if, knowing the reference answer, it's possible to find improvements for the contexts. The metrics that wil be used to perfrom the reference-based datamodels are:

- **ROUGE-L**: This metric measures the longest common subsequence between the generated answer and the reference answer, providing insight into the overlap of content. Any results above 0 will be considered as a positive signal, since the generated answer should be short and concise, and any overlap with the reference answer can be seen as a good sign.

## Results

TODO

## Running the experiments

TODO

## Repository Structure

TODO

