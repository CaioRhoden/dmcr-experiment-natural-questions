python --metric cosine --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_cosine.index
python --metric ip --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_ip.index
python generate_vector_database.py --metric l2 --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_l2.index
python run_experiment --tag l2 
python run_experiment --tag ip
python run_experiment --tag cosine


python generate_random_samples.py