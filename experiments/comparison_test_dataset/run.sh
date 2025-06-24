### GENERATE SEEDS
python run_test.py --run_type setup

## BASELINE GENERARTIONS
python run_test.py --run_type baseline --seed_idx 0
python run_test.py --run_type baseline --seed_idx 1
python run_test.py --run_type baseline --seed_idx 2
python run_test.py --run_type baseline --seed_idx 3
python run_test.py --run_type baseline --seed_idx 4


### NAIVE RAG GENERATIONS
python run_test.py --run_type rag --seed_idx 0
python run_test.py --run_type rag --seed_idx 1
python run_test.py --run_type rag --seed_idx 2
python run_test.py --run_type rag --seed_idx 3
python run_test.py --run_type rag --seed_idx 4


### DATAMODELS GENERATIONS