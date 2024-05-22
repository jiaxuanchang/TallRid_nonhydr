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
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars'],iters=iters)
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'],iters=iters)
ds2 = ds2.chunk(chunks={"XG":372,"XC":372})


xc=ds1.coords['XC']
yc=ds1.coords['YC']

rhoNil = 999.8

# BC-BT conversion
Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[ds2.time.values,yc,xc], dims=['time','YC','XC'])
#print('Conv'+str(Conv))

# kl10 vertical dissipation
dz=200/40 #H/nz
Dsp=np.sum(rhoNil*ds1['KLeps']*dz,axis=1)
print(Dsp)
vDisp=xr.DataArray(Dsp.data, coords=[ds1.time.values,yc,xc], dims=['time','YC','XC'])
print('vDisp'+str(vDisp))


Conv.to_netcdf('../reduceddata/Conv.nc')
vDisp.to_netcdf('../reduceddata/vDisp.nc')





