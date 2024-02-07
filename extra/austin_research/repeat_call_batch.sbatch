#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=single
#SBATCH --job-name=self_repair 
#SBATCH --output=logs/repeat_call_%j.out
#SBATCh --time=24:00:00


python repeat_call.py

