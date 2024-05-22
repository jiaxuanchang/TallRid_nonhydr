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
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['energyvars'],iters=iters)
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'],iters=iters)
ds2 = ds2.chunk(chunks={"XG":372,"XC":372})


xc=ds1.coords['XC']
yc=ds1.coords['YC']

rhoNil = 999.8

# dEbtdt, the rate of baroctropic energy
dtEbt=np.gradient(ds1.SDIAG1.values*rhoNil,dt,axis=0)
print('dtEbc'+str(dtEbt.shape))
dEbtdt = xr.DataArray(dtEbt.data, coords=[ds1.time,yc,xc], dims=['time','YC', 'XC'])

# dEbcdt, the rate of baroclinic energy
Ebc=rhoNil*ds1['SDIAG5']
dtEbc=np.gradient(Ebc,dt,axis=0)
print('dtEbc'+str(dtEbc.shape))
dEbcdt = xr.DataArray(dtEbc.data, coords=[ds2.time.values,yc,xc], dims=['time','YC', 'XC'])


dEbtdt.to_netcdf('../reduceddata/dEbtdt.nc')
dEbcdt.to_netcdf('../reduceddata/dEbcdt.nc')





