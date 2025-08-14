#!/bin/bash
#SBATCH --job-name=llm_as_judge_generation_7270_
#SBATCH --output=/home/caio.rhoden/slurm/llm_as_judge_generation_7270_%j.out
#SBATCH --error=/home/caio.rhoden/slurm/llm_as_judge_generation_7270_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=15G
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

### DATAMODELS RUN ON COLLECTIONS
# python run_experiment.py --run_type datamodels_collections --seed_idx 0 --train_checkpoint 1000 --test_checkpoint 1000 --train_start_idx 10000 --train_end_idx 20000
# python run_experiment.py --run_type datamodels_collections --seed_idx 1
# python run_experiment.py --run_type datamodels_collections --seed_idx 2
# python run_experiment.py --run_type datamodels_collections --seed_idx 3
# python run_experiment.py --run_type datamodels_collections --seed_idx 4

# ### DATAMODELS training
# python run_experiment.py --run_type datamodels_training --seed_idx 0
# python run_experiment.py --run_type datamodels_training --seed_idx 1
# python run_experiment.py --run_type datamodels_training --seed_idx 2
# python run_experiment.py --run_type datamodels_training --seed_idx 3
# python run_experiment.py --run_type datamodels_training --seed_idx 4

# ### DATAMODELS GENERATIONS
python run_experiment.py --run_type datamodels_generations --seed_idx 0 --datamodels_generation_name judge_generation --model_run_id judge_llm
# python run_experiment.py --run_type datamodels_generations --seed_idx 1
# python run_experiment.py --run_type datamodels_generations --seed_idx 2
# python run_experiment.py --run_type datamodels_generations --seed_idx 3
# python run_experiment.py --run_type datamodels_generations --seed_idx 4