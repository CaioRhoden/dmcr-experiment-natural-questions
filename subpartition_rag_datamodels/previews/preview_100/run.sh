

CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s setup
CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s get_rag_retrieval
CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s get_generations
