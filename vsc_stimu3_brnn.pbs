#!/bin/bash

#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=10:gpus=1
cd $PBS_O_WORKDIR

#module load TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4

./run.sh

