# Testing Prompt

The goal here is to evaluate and compare the performance of the models when using two different instructions.  
One retrieved from the paper Power of Noise and the other one is a similar option created for this work.  
The evaluation metrics of the performance will be ROUGE-l and F1Score.  

## What will be evaluated

- Number of non-zero collections

## Setup

Model Llama-3.2-8B-Instruct, 50 test samples with RAG subset of size 100 using coside distance index, 2000 train collections for train and 400 for test. 

Intrustctions:
- **Power of Noise**: 
- **New Prompt**: "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents"