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
    * 05/12/202: Starting today using the dmcr version 0.4.3, some creations of collections and pre_collections of Datamodels may break because of the load of indexes and trian/test dataset isn't automatic anymore. If you want to follow an example of application see the "experiments/debugging_rag_recall_vllm_vs_hfpipeline/validating_datamodels"
    




