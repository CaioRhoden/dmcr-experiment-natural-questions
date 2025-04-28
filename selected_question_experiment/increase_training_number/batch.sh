#!/bin/bash
#SBATCH --job-name=caio_rhoden_taco_task
#SBATCH --output=/home/caio.rhoden/slurm/%j.out
#SBATCH --error=/home/caio.rhoden/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12G
#SBATCH --mail-user=c214129@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

python3 pipeline.py -s setup

python3 pipeline.py -s setter -i 0
python3 pipeline.py -s pre_collections -i 0
python3 pipeline.py -s collections -i 0
python3 pipeline.py -s train -i 0
python3 pipeline.py -s evaluate -i 0

python3 pipeline.py -s setter -i 1
python3 pipeline.py -s pre_collections -i 1
python3 pipeline.py -s collections -i 1
python3 pipeline.py -s train -i 1
python3 pipeline.py -s evaluate -i 1

python3 pipeline.py -s setter -i 2
python3 pipeline.py -s pre_collections -i 2
python3 pipeline.py -s collections -i 2
python3 pipeline.py -s train -i 2
python3 pipeline.py -s evaluate -i 2

python3 pipeline.py -s setter -i 3
python3 pipeline.py -s pre_collections -i 3
python3 pipeline.py -s collections -i 3
python3 pipeline.py -s train -i 3
python3 pipeline.py -s evaluate -i 3


python3 pipeline.py -s setter -i 4
python3 pipeline.py -s pre_collections -i 4
python3 pipeline.py -s collections -i 4
python3 pipeline.py -s train -i 4
python3 pipeline.py -s evaluate -i 4