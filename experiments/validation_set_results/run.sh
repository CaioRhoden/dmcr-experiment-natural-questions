## Run zero shot lllama 3.2 3B
python3 run_zero_shot.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct
python3 run_rag.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct

python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct --run_type setup
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct --run_type pre_collections
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct --run_type collections
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct --run_type training
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/llms/Llama-3.2-3B-Instruct --run_type generation


