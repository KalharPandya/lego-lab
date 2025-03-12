#!/bin/bash
#SBATCH --job-name=MyModelTraining
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=015:00:00

module load anaconda3/2022.05  # Load the Anaconda module
source activate vision            # Activate your conda environment

srun python object_detection_10_epoch_job.py       # Replace with your actual script name
