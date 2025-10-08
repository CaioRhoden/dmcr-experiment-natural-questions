## Run zero shot lllama 3.2 3B
python3 run_zero_shot.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --attn_implementation flash_attention_2
python3 run_zero_shot.py --model gpt-oss-20b --batch_size 8 --language_model_path models/gpt-oss-20b --attn_implementation flash_attention_2
python3 run_zero_shot.py --model DeepSeek-R1-Distill-Qwen-1.5B --batch_size 8 --language_model_path models/DeepSeek-R1-Distill-Qwen-1.5B --attn_implementation flash_attention_2


python3 run_rag.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --attn_implementation flash_attention_2
python3 run_rag.py --model gpt-oss-20b --batch_size 8 --language_model_path models/gpt-oss-20b --attn_implementation flash_attention_2
python3 run_rag.py --model DeepSeek-R1-Distill-Qwen-1.5B --batch_size 8 --language_model_path models/DeepSeek-R1-Distill-Qwen-1.5B --attn_implementation flash_attention_2


python3 run_datamodels.py --model gpt-oss-20b --language_model_path models/gpt-oss-20b --run_type setup
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --language_model_path models/Llama-3.2-3B-Instruct --run_type setup
python3 run_datamodels.py --model DeepSeek-R1-Distill-Qwen-1.5B --language_model_path models/DeepSeek-R1-Distill-Qwen-1.5B --run_type setup


## Recommended parallel execution for the pre-collections and collections stages, use start_idx and end_idx to define the range of samples to be processed in each execution
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type pre_collections --mode train
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type pre_collections --mode test
python3 run_datamodels.py --model gpt-oss-20b --batch_size 8 --language_model_path models/gpt-oss-20b --run_type pre_collections --mode train
python3 run_datamodels.py --model gpt-oss-20b --batch_size 8 --language_model_path models/gpt-oss-20b --run_type pre_collections --mode test
python3 run_datamodels.py --model DeepSeek-R1-Distill-Qwen-1.5B --batch_size 8 --language_model_path models/DeepSeek-R1-Distill-Qwen-1.5B --run_type pre_collections --mode train
python3 run_datamodels.py --model DeepSeek-R1-Distill-Qwen-1.5B --batch_size 8 --language_model_path models/DeepSeek-R1-Distill-Qwen-1.5B --run_type pre_collections --mode test

python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type collections --mode train --collection_id datamodels
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type collections --mode test --collection_id datamodels


python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type training
python3 run_datamodels.py --model Llama-3.2-3B-Instruct --batch_size 8 --language_model_path models/Llama-3.2-3B-Instruct --run_type generation