#!/bin/bash --login
###
#job name
#SBATCH --job-name=translate
#job stdout file
#SBATCH --output=translate.out.%J
#job stderr file
#SBATCH --error=translate.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=0-10:00
#memory per process in MB
#SBATCH --mem-per-cpu=9000
#run a single task, using a single CPU core
#SBATCH --ntasks=4
#tasks to run per node
#SBATCH --ntasks-per-node=4
###
python3 T2.py
