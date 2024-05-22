#!/bin/zsh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:35:30
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N mdstonetcdf

cd $PBS_O_WORKDIR
source /u/home/jklymak/.zshrc
echo ${PATH}
for PRE in LWT1kmlowU0T025Amp050f100
do
  /p/home/jklymak/miniconda3/bin/python GetEnergyBudgetNew.py $PRE 0
done

rsync -av ../reduceddata/*.nc pender.seos.uvic.ca:AbHillTide/reduceddata/
