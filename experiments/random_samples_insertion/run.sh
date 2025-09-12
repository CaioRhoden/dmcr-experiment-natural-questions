python generate_random_samples.py

python3 run_rag.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct

python3 run_datamodels.py --language_model_path models/Llama-3.2-3B-Instruct --run_type pre_collections --start_idx 0 --end_idx 1000 --checkpoint 50 --mode train
python3 run_datamodels.py --language_model_path models/Llama-3.2-3B-Instruct --run_type pre_collections --start_idx 0 --end_idx 1000 --checkpoint 50 --mode test
python3 run_datamodels.py --language_model_path models/Llama-3.2-3B-Instruct --run_type collections --start_idx 0 --end_idx 500000 --checkpoint 5000 --mode train
python3 run_datamodels.py --language_model_path models/Llama-3.2-3B-Instruct --run_type collections --start_idx 0 --end_idx 500000 --checkpoint 5000 --mode test

python3 run_datamodels.py --language_model_path models/Llama-3.2-3B-Instruct --run_type training --model_run_id judge_datamodel --collection_id random_samples_insertion_20