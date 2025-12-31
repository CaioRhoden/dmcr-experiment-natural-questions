#!/bin/bash
#SBATCH --job-name=bge_ip
#SBATCH --output=/home/caio.rhoden/slurm/%j_bge_ip.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_bge_ip.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq

python create_indeces.py --indice_name bge