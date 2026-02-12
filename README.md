# dmcr-experiment-natural-questions

This repo is an experiment of the application of Datamodels as a RAG tool for the LLM Benchmark Natural Questions
This is derived from the previous work on: https://github.com/CaioRhoden/datamodels-context-reduction

Important to mention, the Natural Questionss benchmmark can be found here: https://github.com/google-research-datasets/natural-questions
Part the experiment consists in validate the influence of context, the gold documents are retrieved of the previous study and dataset from: https://github.com/florin-git/The-Power-of-Noise

## Requirements
    * Envinronment Installation: ```pip install -r requirements.txt```
    * FAISS Instalation (unsing Conda): ```conda install -c pytorch faiss-gpu``` (The CPU option is possible with pip)
    * ```pip install -e .```


## Logs
    * 05/12/2025: Starting today using the dmcr version 0.4.3, some creations of collections and pre_collections of Datamodels may break because of the load of indexes and trian/test dataset isn't automatic anymore. If you want to follow an example of application see the "experiments/debugging_rag_recall_vllm_vs_hfpipeline/validating_datamodels"

    * 08/01/2026: Using the v0.4.4 version of dmctr lib, fizing the evaluation of datamodels models with multiple weights files

    * 10/02/2026: Version 0.4.5 from dmcr, fixing the "format_input" callable function for pre-collection creation, before it was using only training which was inserting wrong distributions for the tests sets. It impacted the training datamodels evaluation to understand the quality of the training.
    




