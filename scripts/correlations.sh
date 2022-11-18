#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=correlations
#SBATCH --account=rrg-wperciva
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G
python get_correlations.py