python generate_random_samples.py
python3 run_rag.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct



python3 run_datamodels.py --run_type collections --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --start_idx 0 --end_idx 500000 --checkpoint 5000 --collection_id groundtruth_random_20