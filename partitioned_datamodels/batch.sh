#!/bin/bash
#SBATCH --job-name=caio_rhoden_taco_task
#SBATCH --output=/home/caio.rhoden/slurm/%j.out
#SBATCH --error=/home/caio.rhoden/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --mail-user=c214129@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
python3 pipeline.py -s setter -i 0
python3 pipeline.py -s pre_collections -i 0
python3 pipeline.py -s pre_collections -i 1
python3 pipeline.py -s pre_collections -i 2
python3 pipeline.py -s pre_collections -i 3
python3 pipeline.py -s setter -i 4
python3 pipeline.py -s pre_collections -i 4