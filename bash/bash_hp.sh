#!/bin/bash

#SBATCH --job-name=hp_tr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:p40:4
#SBATCH --mail-type=END
#SBATCH --mail-user=cs2737@nyu.edu
  
module purge
module load python3/intel/3.6.3

python3 harry_potter_train.py
