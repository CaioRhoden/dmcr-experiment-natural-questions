# python --metric cosine --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_cosine.index
# python --metric ip --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_ip.index
# python generate_vector_database.py --metric l2 --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_l2.index

# python generate_random_samples.py
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python run_experiment.py --tag l2 
python run_experiment.py --tag ip
python run_experiment.py --tag cosine


