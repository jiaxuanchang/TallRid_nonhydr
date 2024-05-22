#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=julia27317@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=3G
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o runSaveData.sh.log-%j

echo "Working Directory = $(pwd)"

source ${HOME}/ENV/bin/activate

python Savedata_11T.py  > SaveData.out

rsync -av --stats ../reduceddata/*.nc jxchang@cedar.computecanada.ca:/Users/jiaxuanchang/Documents/PHD/KnightInlet/ModelExs/HighRes2/upload by sync sftp/TideU008N0LinH200ho140Ah0200Kh0200Cdqdt003/reduceddata/
