# dmcr-experiment-natural-questions

## Requirements
    * Envinronment Installation: ```pip install -r requirements.txt```
    * FAISS Instalation (unsing Conda): ```conda install -c pytorch faiss-gpu``` (The CPU option is possible with pip)

# dcr-experiment-nq
This repo is an experiment of the application of Datamodels as a RAG tool for the LLM Benchmark Natural Questions
This is derived from the previous work on: https://github.com/CaioRhoden/datamodels-context-reduction

Important to mention, the Natural Questionss benchmmark can be found here: https://github.com/google-research-datasets/natural-questions
Part the experiment consists in validate the influence of context, the gold documents are retrieved of the previous study and dataset from: https://github.com/florin-git/The-Power-of-Noise
