#!/bin/bash
#SBATCH --job-name=generating_index
#SBATCH --output=/home/caio.rhoden/slurm/%j_generating_index.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_generating_index.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=45G
#SBATCH --time=14:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx8000

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# python generate_vector_database.py --metric cosine --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_cosine_upgrade.index
# python generate_vector_database.py --metric ip --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_ip_upgrade.index
# python generate_vector_database.py --metric l2 --saving_path ../../data/wiki_dump2018_nq_open/processed/wiki_l2_upgrade.index

# python generate_random_samples.py


python run_experiment.py --tag l2 
python run_experiment.py --tag ip
# python run_experiment.py --tag cosine


