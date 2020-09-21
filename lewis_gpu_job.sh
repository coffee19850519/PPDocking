#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH -A xulab
#SBATCH -p BioCompute,Lewis         # use the BioCompute partition
#SBATCH -J merge_regression_pt  #job's name
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem-per-cpu=128G
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

## Module Commands
module load miniconda3

## Activate your Python Virtual Environment (if needed)
source activate /storage/hpc/data/hefe/.conda/envs/my_environment

# Science goes here:
python /storage/htc/joshilab/hefe/docking/protein_quality_assessment_code/graph_creation/regression_merge.py

source deactivate

echo "### Ending at: $(date) ###"





