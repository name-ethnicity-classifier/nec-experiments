#!/bin/bash

#SBATCH -A HOUTEN-SL3-GPU
#SBATCH -p pascal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=240

python train_model.py

# current job: 32414812
