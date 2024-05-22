from xmitgcm import open_mdsdataset
import xgcm
import os
import numpy as np
import xarray as xr
import math


# get to the input folder (where MITgcm outputs)
currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)


# load data to be xarray
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars'],iters=10)
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)

Depth=ds1.Depth

Depth.to_netcdf('../reduceddata/Depth.nc')

