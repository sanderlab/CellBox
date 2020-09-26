#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p short
#SBATCH --mem=10g                         # Memory total in MB (for all cores)
#SBATCH -o slurm_out/%j.out
#SBATCH -e slurm_out/%j.err

source ~/.bash_profile

module load gcc/6.2.0
module load python/3.7.4
source $labhome/bo/venv37/bin/activate
cd $labhome/bo/CellBox/CellBox

for i in {000..099}
    do
        python scripts/main.py -config="${1}" -i=$i
    done

