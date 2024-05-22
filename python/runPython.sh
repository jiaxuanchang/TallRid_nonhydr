#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=julia27317@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=3G
#SBATCH --time=0:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o runPython.sh.log-%j

echo "Working Directory = $(pwd)"

source ${HOME}/ENV/bin/activate

python GetTemp.py
python GetDepth.py
python GetUVW.py
python GetBottomDrag.py
python GetConvVDisp.py
python GetdEdt.py
python GetHDissip.py
python GetRadiation.py
python GetZeta.py
python GetKLeps.py
python GetPHIHYD.py
python GetPHIBOT.py
#python cal_myEnergyBudget.py
