python --metric cosine --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_cosine.index
python --metric ip --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_ip.index
python --metric l2 --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_l2.index

python generate_random_samples.py