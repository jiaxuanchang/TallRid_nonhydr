#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=julia27317@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=3G
#SBATCH --time=0:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o runSaveTideavEnergy.sh.log-%j

echo "Working Directory = $(pwd)"

source ${HOME}/ENV/bin/activate

python -u Save_tidalavEnergyBudget.py > Save_tidalEnergyBudget.out
python -u Save_tidalavFlux.py > Save_tidalFlux.out

