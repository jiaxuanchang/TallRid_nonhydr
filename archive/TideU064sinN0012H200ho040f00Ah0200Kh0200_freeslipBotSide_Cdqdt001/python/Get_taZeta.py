from xmitgcm import open_mdsdataset
import xgcm
import os
import numpy as np
import xarray as xr
import math
import pandas as pd

# get to the input folder (where MITgcm outputs)
currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)


# load data to be xarray
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars'])
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"time":24,"YG":30,"YC":30,"XG":372,"XC":372})  # make chunks in x smaller
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
zetachunked=zeta.chunk(chunks={"time":24,"YG":30,"XG":372})
print(zetachunked.chunks)
sm_zeta = zetachunked.rolling(XG=windowsize,center=True).mean()
sm_zeta = sm_zeta.rolling(YG=windowsize,center=True).mean()
print(sm_zeta)
#print(sm_zeta.isel(XG=720,YG=60,time=20).values)

# time bin as M2 tidal cycle
time=ds1.coords["time"].values/np.timedelta64(1, 's')
ttlen=len(ds1.time)
T=12.4*3600
print('ttlen:'+str(ttlen))
print('time[-1]:'+str(time[-1]))
time_bin_labels = np.arange(12.4*60*60/2,time[-1]-20000,12.4*60*60)
print('time_bin_labels:' +str(time_bin_labels))

print(time[-1]/T)
nT=math.floor(time[-1]/T)

time_bin = pd.timedelta_range(0, periods=nT+1,freq='44660S')
print('time_bin:' + str(time_bin))
time_ns = pd.timedelta_range(0, periods=ttlen,freq='1860S')

ta_sm_zeta = sm_zeta.groupby_bins('time',time_bin,labels=time_bin_labels).mean()


#ds = xr.Dataset(coords={"XG": xg,
#                        "rAz":ds1.rAz,
#                        "YG":yg,
#                        "Z":ds1.Z,
#                        "time":ds1.time,
#                        "dyZ":dyZ,
#                        "maskZ":maskZ,
#                        "drF":ds1.drF}) 

#ds.to_netcdf('./reduceddata/sm_zeta.nc') seems to be wrong coords when read locally I think problem is cedar with engine:netcdf4,h5netcdf
ta_sm_zeta.to_netcdf('../reduceddata/ta_sm_zeta.nc')




