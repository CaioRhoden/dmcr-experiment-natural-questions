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


### DATAMODELS RUN ON PRE-COLLECTIONS
python run_test.py --run_type datamodels_pre_collections --seed_idx 0
python run_test.py --run_type datamodels_pre_collections --seed_idx 1
python run_test.py --run_type datamodels_pre_collections --seed_idx 2
python run_test.py --run_type datamodels_pre_collections --seed_idx 3
python run_test.py --run_type datamodels_pre_collections --seed_idx 4

### DATAMODELS RUN ON COLLECTIONS
python run_test.py --run_type datamodels_collections --seed_idx 0
python run_test.py --run_type datamodels_collections --seed_idx 1
python run_test.py --run_type datamodels_collections --seed_idx 2
python run_test.py --run_type datamodels_collections --seed_idx 3
python run_test.py --run_type datamodels_collections --seed_idx 4

### DATAMODELS training
python run_test.py --run_type datamodels_training --seed_idx 0
python run_test.py --run_type datamodels_training --seed_idx 1
python run_test.py --run_type datamodels_training --seed_idx 2
python run_test.py --run_type datamodels_training --seed_idx 3
python run_test.py --run_type datamodels_training --seed_idx 4

### DATAMODELS GENERATIONS
python run_test.py --run_type datamodels_generations --seed_idx 0
python run_test.py --run_type datamodels_generations --seed_idx 1
python run_test.py --run_type datamodels_generations --seed_idx 2
python run_test.py --run_type datamodels_generations --seed_idx 3
python run_test.py --run_type datamodels_generations --seed_idx 4