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

# assign time index to read from tidal period number
iT=range(6,8)
P=12.4*60*60
dt=1860
t_st=int(P*iT[0])
t_en=int(P*(iT[-1]))
iters=range(t_st,t_en,dt)
print(iters)

# load data to be xarray
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars','energyvars'],iters=iters)
#print(ds1.chunks)
#ds1 = ds1.chunk(chunks={"time":4,"YG":30,"YC":30,"XG":360,"XC":360})  # make chunks in x smaller
#print(ds1.chunks)


xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']

ds1.UVEL.to_netcdf('../reduceddata/UVEL_%s.nc' %iters)
ds1.VVEL.to_netcdf('../reduceddata/VVEL_%s.nc' %iters)
ds1.WVEL.to_netcdf('../reduceddata/WVEL_%s.nc' %iters)




