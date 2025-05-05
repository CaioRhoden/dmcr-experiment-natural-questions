# Subpartition RAG Datamodels Experiment

## Goal
Test if the datamodels can achieve a better selection of context over a subpartition of selection from traditional RAG, it would work as post-processing tool of the RAG with the goal of identify detractor and possibly find more diverse contexts

## Setup

The experiment here is with the Natural Questions dataset, especially using the Gold Documents. The total size of the test set consists in 2889 samples with question, target answers and from exactly what passage from the Wikipedia dump 2018 the answer is retrieved.

## Methodology

The experiment pipeline will follow some steps:
1. Generate three different index for the wikipedia dump 2018 with FAISS, one for each of the following metrics: INNER PRODUCT and L2 (Euclidean Distance)
2. For each sample of the test set get a .json containing the top 100 closest samples with the similarity score.
3. Create generations for different context-window sizes: 4, 8 and 16
4. Run a datamodels over them, varying the parameters (TODO SPECS) to find through the dataset the differences in the contexts and the generation.
5. Compute the evaluation considering the performance metric  Rouge-L achieved and the # of context windows with the GOLD DOCUMENT
