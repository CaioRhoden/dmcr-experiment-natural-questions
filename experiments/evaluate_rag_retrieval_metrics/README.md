# Evaluate RAG Retrieval Metrics

The objective of this experiment is to evaluate the RAG method in a hablation of random 250 samples from the validation dataset. 


## Methodology

The model that will be used is the LLama-3.2-3B with k=16 for window context with temperature 0.7 and top_p=0.9. The embedder being used is the "bge-base-en-v1.5". The metrics that will be evaluated are:
- FAISS L2
- FAISS IP
- Chroma COSINE DISTANCE

### Requirements
* Wikipedia 2018 dump indexed for each metric
* Random set example of 250 samples from validation set


### Questions
* How does each RAG perform?

### Decisions
* Which LLM is to be used?
* What should import from other experiments?
* **R1:** LLAMA 3.2 - Instruct (Maybe MISTRAL-7B)
* **R2:** Datasets retriever
    * Hallucination dataset
    * Generation and Datasets retriever (*Note or script*)