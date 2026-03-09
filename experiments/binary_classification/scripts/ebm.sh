#!/bin/bash
#SBATCH --job-name=training_ebm_voting_pipeline
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_training_ebm_voting_pipeline.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_training_ebm_voting_pipeline.err
#SBATCH --cpus-per-task=11
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq


cd ebm
python voting_ebm_pipeline.py