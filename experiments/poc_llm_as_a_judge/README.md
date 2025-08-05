# PoC: LLM as a Judge

The objective of this experiment is to evaluate a re-ranking RAG method with Datamodels using the LLM as judge single grading apporach from the paper [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).


## Methodology

The training of the datamodels will be based on the preivous experiment *datamodels_training_window_size*, the same collections dataset will be used with the same type of paramters setup. The size of the context will be of **k=16** with the reuslts being compared with the training size of 2000 collections. To be able to compute that the datamodels structure will be replicated here.

### Requirements
* Generation of normalized score.
* Comparison between Weir's RAG & Datasets & Judge.

### Questions
* How does the LLM as judge perform?
* What is the distribution of scores per random seed?
* What is the context best suited overlap with other techniques?

### Decisions
* Which LLM is to be used?
* What should import from other experiments?
* **R1:** LLAMA 3.2 - Instruct (Maybe MISTRAL-7B)
* **R2:** Datasets retriever
    * Hallucination dataset
    * Generation and Datasets retriever (*Note or script*)