Code to generate the simulations reported in "Saturation of Internal Tide Generation over Shallow Supercritical Topography" by Chang, J-X and Klymak, J. M.

# Some Runs on CEDAR.computecanada.ca

## Contents:

  - `MITgcm66h` is my version with `NF90io`.
  - `input` is where most model setup occurs.
  - `python` is where most processing occurs.

## Vagaries

  - cannot seem to get netcdf to work
  - work in `project/jxchang/`.  I do a softlink: `ln -s /home/jxchang/project/jxchang/TallRid_nonhydr TallRid_nonhydr`.  
  - link 'results/' to `/home/jxchang/scratch/TallRid_nonhydr`

## To compile on Cedar

  - `module load gcc openmpi`
  - `cd build/`
  - `../MITgcm66h/tools/genmake2 -optfile=../build_options/cedargcc -mods=../code/ -rootdir=../MITgcm66h --mpi`
  - OPTIONAL: `make CLEAN` to make sure no bad compiled files are left over
  - `make depend`
  - `make`

## To run

  - run `python gendata.py`
  - run `sbatch -J jobname runModel.sh` where "jobname" is the directory where you put the "results"
  - run `sq` to see what progress is...
