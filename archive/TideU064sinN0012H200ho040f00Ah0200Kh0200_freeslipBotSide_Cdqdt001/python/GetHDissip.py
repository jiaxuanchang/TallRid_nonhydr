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
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})  # make chunks in x smaller
print(ds1.chunks)

# use xgcm grid tech to have easier interp
grid = xgcm.Grid(ds1, periodic=False)
print(grid)

xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
time=ds1.coords['time']

# depth mean velocity
U0W=((ds1['UVEL']*ds1['drF']*ds1['hFacW']).sum('Z'))/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
V0S=((ds1['VVEL']*ds1['drF']*ds1['hFacS']).sum('Z'))/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
print('U0W')
print(U0W)
ds1['U0W']=xr.DataArray(U0W.data,coords=[ds1.time,yc,xg],dims=['time','YC','XG'])
ds1['V0S']=xr.DataArray(V0S.data,coords=[ds1.time,yg,xc],dims=['time','YG','XC'])
del U0W,V0S

# baroclinic velocity
ds1['upW']=(ds1['UVEL']-ds1['U0W'])*ds1['maskW']
ds1['vpS']=(ds1['VVEL']-ds1['V0S'])*ds1['maskS']
ds1['maskL']=grid.interp(ds1.hFacC,'Z',to='left', boundary='extrapolate')
wp=ds1['WVEL'].where(ds1['maskL'] !=0, np.nan)

# calculate horizontal dissipation
# setup other masks and hFac and dr
# for dvdx
ds1['maskZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate')
ds1['hFacZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate')
# for dwdx
ds1['hFacWL'] = grid.interp(ds1.hFacW, 'Z', to='left', boundary='extrapolate')
print(ds1.hFacWL.dims)
# for dwdy
ds1['hFacSL'] = grid.interp(ds1.hFacS, 'Z', to='left', boundary='extrapolate')
print(ds1.hFacSL.dims)
ds1['drL']=grid.interp(ds1.drF, 'Z', to='left', boundary='extrapolate')

# separately calculate because grids are different
# dudx & dvdy at (xc,yc)
rhoNil = 999.8
viscAh = 2.e-2
hDispbc1 = rhoNil*viscAh*(((grid.diff(ds1.upW* ds1.dyG , 'X', boundary='extrapolate')/ds1.rA)**2
                           #+(grid.diff(ds1.vpS* ds1.dxG , 'Y', boundary='extrapolate')/ds1.rA)**2
                           )\
                           *(ds1['drF']*ds1['hFacC'])).sum('Z') 
print('dudx')
print(hDispbc1)
# dvdx & dudy at (xg,yg)
hDispbc2 = rhoNil*viscAh*(((grid.diff(ds1.vpS* ds1.dyC , 'X', boundary='extrapolate')/ds1.rAz)**2
                           #+(grid.diff(ds1.upW* ds1.dxC , 'Y', boundary='extrapolate')/ds1.rAz)**2
                           )\
                           *(ds1['drF']*ds1['hFacZ'])).sum('Z')
dvdx=xr.DataArray(hDispbc2.values, coords=[time,yc, xc], dims=["time","YC", "XC"])
# dwdx at (xg,yc)
hDispbc31 = rhoNil*viscAh*(((grid.diff(wp ,'X',boundary='extrapolate')/ds1.dxC)**2)\
                          *(ds1['drL']*ds1['hFacWL'])).sum('Zl')
print('dwdx')
print(hDispbc31)
print(grid.interp(hDispbc31,'X',boundary='extrapolate'))
# dwdy at (xc,yg)
hDispbc32 = 0#rhoNil*viscAh*(((grid.diff(wp ,'Y',boundary='extrapolate')/ds1.dyC)**2)\
                          #*(ds1['drL']*ds1['hFacSL'])).sum('Zl')


hDispbc = hDispbc1+dvdx+grid.interp(hDispbc31,'X',boundary='extrapolate') 
                        #grid.interp(hDispbc2,['X','Y'],boundary='extrapolate')\
                        #+grid.interp(hDispbc32,'Y',boundary='extrapolate')



print(hDispbc)

hDispbc.to_netcdf('../reduceddata/hDisp.nc')





