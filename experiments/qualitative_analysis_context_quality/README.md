# Qualitative Analysis Context Quality

The objective of this experiment is to evaluate a re-ranking RAG method with Datamodels using the LLM as judge single grading apporach from the paper [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).


## Methodology
The goal of this experiment is to further understand the relations between the context generated using RAG and datamodels. To do so different perspectives must be analysed:

### Global metrics
- **Number of gold documents on k**:
- **Context intersection in top k documents**:
- **Average syntax similarity with response**
- **Average max ranking for datamodels**


### Showcases
- **Rag-to-datamodels index**




### Requirements


### Questions


### Decisions
* Which LLM is to be used?
* What should import from other experiments?
* **R1:** LLAMA 3.2 - Instruct (Maybe MISTRAL-7B)
* **R2:** Datasets retriever
    * Hallucination dataset
    * Generation and Datasets retriever (*Note or script*)