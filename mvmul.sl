#!/bin/bash -l
#SBATCH --job-name=omp_cpu
#SBATCH --account=def-ehyangit # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-0:05:00         # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8      # adjust this if you are using parallel commands
#SBATCH --mem-per-cpu= 2000            # adjust this according to your the memory requirement per node you need


gcc -O3 -march=native -ffast-math  -msse4 -w -lstdc++ -o loop_conv -fopenmp loop_conv.cpp

gcc -O3 -march=native -ffast-math  -msse4 -w -lstdc++ -o loop_conv_last -fopenmp loop_conv_last.cpp

time ./loop_conv >loop_conv.txt

timr ./loop_conv_last >loop_conv_last.txt