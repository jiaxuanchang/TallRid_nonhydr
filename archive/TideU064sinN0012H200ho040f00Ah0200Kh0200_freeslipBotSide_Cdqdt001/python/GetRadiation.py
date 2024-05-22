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
iT=range(6,12)
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

grid = xgcm.Grid(ds1, periodic=False)
print(grid)

xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']

rhoNil = 999.8

# hdFbt
Fxbt=ds2.SDIAG2+ds2.SDIAG4   
Fybt=ds2.SDIAG3              # should be vPbt + vEbt but no vEbt in diagnostics 
Fxbt=xr.DataArray(rhoNil*Fxbt.values, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
Fybt=xr.DataArray(rhoNil*Fybt.values, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
hd_Fbt=(grid.diff(Fxbt*ds2['dyG'],'X',boundary='extrapolate'))/ds2['rA']
#hd_Fbt=(grid.diff(Fxbt*ds2['dyG'],'X',boundary='extrapolate')+grid.diff(Fybt*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']
print(hd_Fbt)

# hdFbc
uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
#print(uPbc)
#print(uEbc)
Fxbc=uPbc+uEbc
Fybc=vPbc+vEbc
FxbcC=grid.interp(Fxbc,'X', boundary='extrapolate') 
#FybcC=grid.interp(Fybc,'Y', boundary='extrapolate') 
#print(Fbc)
hd_Fbc=(grid.diff(Fxbc*ds2['dyG'],'X',boundary='extrapolate'))/ds2['rA']
#hd_Fbc=(grid.diff(Fxbc*ds2['dyG'],'X',boundary='extrapolate')+grid.diff(Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']
print(hd_Fbc)


Fxbc.to_netcdf('../reduceddata/Fxbc_%s.nc' %iters)
FxbcC.to_netcdf('../reduceddata/FxbcC_%s.nc' %iters)
Fybc.to_netcdf('../reduceddata/Fybc_%s.nc' %iters)
#FybcC.to_netcdf('../reduceddata/FybcC_%s.nc' %iters)
hd_Fbt.to_netcdf('../reduceddata/hd_Fbt_%s.nc' %iters)
hd_Fbc.to_netcdf('../reduceddata/hd_Fbc_%s.nc' %iters)





