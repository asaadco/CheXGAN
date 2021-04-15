#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J jupyter
#SBATCH -o jupyter.%J.out
#SBATCH -e jupyter.%J.err
#SBATCH --mail-user=asaad.alghamdi.1@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=02:30:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]

#run the application:
srun --resv-ports=1 bin/launch-jupyter-server.srun
