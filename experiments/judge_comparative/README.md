# Judge Comparative Experiment

The goal of the experiment is to evaluate how close each reference-free binary classification judge is close to a reference-based metric (ROUGE-L) in terms of ROC-AUC Score.  


## Results


## Setup
1. Creation of a random subset of 1000 samples from the NQ dev set through the `subset_generation.py` script.
2. Generation of the ZeroShot generations through the `zeroshot_generation.py` script.
3. Generation of the RAG generations through the `rag_generation.py` script.
4. Generation of the pre-collections for the datamodels (where the judges will be applied) through the `pre_collection_generation.py` script.
5. Calculation of the ROUGE-L scores for each generation through the `calculate_rougel.py` script.
6. Computing and saving of the judges scores for each generation through the `judge_scores_calculation.py` script, saving the collections on the `collections/` directory.
7. Analysis of the results through the `analysis.ipynb` notebook on the `analysis/` directory.

## Judges

We will be using the following judges for the experiment:
1. **Simple Judge**: Adaption of the classic LLM-as-a-judge approach to evaluate a generation between GOOD (1) and BAD (0).
2. **Recall Simple Judge**: Adaption of the classic LLM-as-a-judge approach to evaluate a generation between REPONDS (1) and NOT RESPONDS (0). Being more focused on recall, this judge should be more lenient to the generations, giving a higher score to generations that contain the answer but also some hallucinated information.
3. **Zero Shot Pairwise Judge**: Pairwise comparison between two generations, the Zero Shot and the present in the pre-collections, indicating 1 if the pre-collection generation is better than the Zero Shot one and 0 otherwise.
4. **RAG Pairwise Judge**: Pairwise comparison between two generations, the RAG and the present in the pre-collections, indicating 1 if the pre-collection generation is better than the RAG one and 0 otherwise.
5. **Faithfulness Judge**: Judge that evaluates the faithfulness of a generation to the question and the retrieved passage, indicating 1 if the generation is faithful and 0 otherwise.

## Reference Prompts
 
 The prompts used for the judges are available on `judge_prompts.json` file. The instructions for the generation of the pre-collections (generations used to create the judge scores) are the following:

 - *Experiment 1*: You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens
 - *Experiment 2*: You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES.

