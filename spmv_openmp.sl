#!/bin/bash -l
#SBATCH --job-name=omp_cpu
#SBATCH --account=def-ehyangit # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-1:00:00         # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8      # adjust this if you are using parallel commands
#SBATCH --mem-per-cpu=2G            # adjust this according to your the memory requirement per node you need

time python ../OUTPUTs/run_all.py > run_all.txt
