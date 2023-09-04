#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=2500G
#SBATCH --time=24:00:00
#SBATCH --account=rrg-eros-ab
#SBATCH --output=/home/tgw/scratch/slurm-%j.log
#SBATCH --mail-user=thomas.williams@physics.ox.ac.uk
#SBATCH --mail-type=ALL

# Load in modules, activate environment
module load StdEnv/2020
module load gcc/11.3.0
module load python/3.9

source /home/tgw/projects/rrg-eros-ab/tgw/jwst_env/bin/activate

cd /home/tgw/projects/rrg-eros-ab/tgw/jwst_scripts/config/2128

python3 run_reprocessing_long.py

deactivate

#SBATCH --cpus-per-task=24
#SBATCH --mem=125G
#SBATCH --time=1:00:00

#SBATCH --cpus-per-task=32
#SBATCH --mem=2500G
#SBATCH --time=72:00:00

# For the download step
# salloc --time=10:0:0 --mem=8G --ntasks=1 --account=rrg-eros-ab
# salloc --time=1:0:0 --mem=32G --ntasks-per-node=1 --cpus-per-task=16 --account=rrg-eros-ab
