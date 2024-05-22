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
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"time":4,"YG":30,"YC":30,"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)

# use xgcm grid tech to have easier interp
grid = xgcm.Grid(ds1, periodic=False)
print(grid)

xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']

maskZ = grid.interp(ds1.hFacS, 'X', boundary='extend')
dyZ=grid.interp(ds1.dyC,'X',boundary='extrapolate')
zeta = (-grid.diff(ds1.UVEL * ds1['dxC'], 'Y', boundary='extend') + grid.diff(ds1.VVEL * ds1['dyC'], 'X',boundary='extend'))/ds1['rAz']
print('ZETA')
print(zeta)
print(zeta.chunks)
print(min(zeta.chunks))

# smooth vortivity
windowsize=5
zetachunked=zeta.chunk(chunks={"time":4,"YG":30,"XG":360})
print(zetachunked.chunks)
sm_zeta = zetachunked.rolling(XG=windowsize,center=True).mean()
#sm_zeta = sm_zeta.rolling(YG=windowsize,center=True).mean()
print(sm_zeta)
print(sm_zeta.isel(XG=720,time=20).values)

#ds = xr.Dataset(coords={"XG": xg,
#                        "rAz":ds1.rAz,
#                        "YG":yg,
#                        "Z":ds1.Z,
#                        "time":ds1.time,
#                        "dyZ":dyZ,
#                        "maskZ":maskZ,
#                        "drF":ds1.drF}) 

#ds.to_netcdf('./reduceddata/sm_zeta.nc') seems to be wrong coords when read locally I think problem is cedar with engine:netcdf4,h5netcdf
sm_zeta.to_netcdf('../reduceddata/sm_zeta_%s.nc'%iters)
maskZ.to_netcdf('../reduceddata/maskZ.nc')
ds1.drF.to_netcdf('../reduceddata/drF.nc')
dyZ.to_netcdf('../reduceddata/dyZ.nc')




