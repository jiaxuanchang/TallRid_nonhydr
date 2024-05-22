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
P=12.4*60*60   # M2 tide =12.4 hr
dt=1860        # model output timestep
t_st=int(P*iT[0]) 
t_en=int(P*(iT[-1]))
iters=range(t_st,t_en,dt)
print(iters)

# load data to be xarray
ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars'],iters=iters)
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)

# use xgcm grid tech to have easier interp
grid = xgcm.Grid(ds1, periodic=False)
print(grid)

# get bottom velocity
bottom_ind = xr.DataArray(np.floor(ds1.hFacC.sum('Z')).values.astype(int),dims=("YC", "XC"))
print('bottom_ind')
print(bottom_ind)
# sum of hFac = 0 -> wall
bottom_mask=bottom_ind.where(bottom_ind==0,1)
# sum of hFac outside wall is from 1 to 40, but index should be from 0-39
bottom_ind=(bottom_ind-1)*bottom_mask
print(bottom_ind)

# get the central point velocity in a grid
UC=grid.interp(ds1.UVEL,'X', boundary='extrapolate') 
#VC=grid.interp(ds1.VVEL,'Y', boundary='extrapolate') 
wC = grid.interp(ds1.WVEL,'Z', boundary='extrapolate') 
print('UC')
print(UC)

# vectorized indexing
Ubot=UC.isel(Z=bottom_ind)
#Vbot=VC.isel(Z=bottom_ind)
wbot=wC.isel(Z=bottom_ind)

# calculate bottom drag 
Cd=1e-3
rhoNil=999.8
U0C=((UC*ds1['drF']*ds1['hFacC']).sum('Z'))/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')  #barotropic velocity
#V0C=((VC*ds1['drF']*ds1['hFacC']).sum('Z'))/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')
upCbot=(Ubot-U0C)  #baroclinic bottom velocity
#vpCbot=(Vbot-V0C)
absuH=(Ubot**2)**0.5  #barotropic velocity magnitide
D0=rhoNil*Cd*absuH*(Ubot*U0C)  #|U|*(uU+vV)
Dp=rhoNil*Cd*absuH*(Ubot*upCbot+wbot*wbot)  #|U|*(u u'+v v'+w^2)
print(Dp)


D0.to_netcdf('../reduceddata/BTBottomDrag_2.nc')
Dp.to_netcdf('../reduceddata/BCBottomDrag_2.nc')

