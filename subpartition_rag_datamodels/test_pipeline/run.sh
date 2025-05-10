# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s setup
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s get_rag_retrieval
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s get_rag_generations
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s create_datamodels_datasets
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s run_pre_collections
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s run_collections
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s train_datamodels
# CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s  evaluate_datamodels
CUDA_VISIBLE_DEVICES=1 python3 pipeline.py -s get_datamodels_generations

