#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 5
#SBATCH --gres gpu:1
#SBATCH --time=1:00:00

python3 train_gan_model.py -l sinkhorn -l energy -l gaussian -l laplacian --input_dir mock_data