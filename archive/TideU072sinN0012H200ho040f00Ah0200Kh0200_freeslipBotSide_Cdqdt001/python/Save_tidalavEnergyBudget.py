from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import pandas as pd
import numpy as np
import xarray as xr
import math


currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars','energyvars'])
print(ds1.chunks)
ds1 = ds1.chunk(chunks={"XG":372,"XC":372})
print(ds1.chunks)

ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
ds2 = ds2.chunk(chunks={"XG":372,"XC":372})

grid = xgcm.Grid(ds1, periodic=False)
print(grid)

## integrated area ## MAKE SURE ##
xmin = 33000
xmax = 87000
ymin = 0
ymax = 3000
numcolt=21
numcolv=21


time1=ds1.coords['time'].values/np.timedelta64(1, 's')
time=ds2.coords['time'].values/np.timedelta64(1, 's')
xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
z=ds1.coords['Z']
ix=[i for i, e in enumerate(xc) if (e > xmin) & (e < xmax)]
iy=[i for i, e in enumerate(yc) if (e > ymin) & (e < ymax)]
print(ix[0],ix[-1])
print(iy[0],iy[-1])



viscAh=2.e-2  #edit
rhoNil=999.8
T=12.4*3600


ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
dt=time[1]-time[0]
print('dt:' +str(dt))


time_bin_labels = np.arange(12.4*60*60/2,time[-1]-20000,12.4*60*60)
print('time_bin_labels:' +str(time_bin_labels))

print(time[-1]/T)
nT=math.floor(time[-1]/T)

time_bin = pd.timedelta_range(0, periods=nT+1,freq='44660S')
print('time_bin:' + str(time_bin))
time_ns = pd.timedelta_range(0, periods=ttlen,freq='1860S')


# depth mean velocity
U0W=((ds1['UVEL']*ds1['drF']*ds1['hFacW']).sum('Z'))/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
V0S=((ds1['VVEL']*ds1['drF']*ds1['hFacS']).sum('Z'))/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
ds1['U0W']=xr.DataArray(U0W.data,coords=[time_ns,yc,xg],dims=['time','YC','XG'])
ds1['V0S']=xr.DataArray(V0S.data,coords=[time_ns,yg,xc],dims=['time','YG','XC'])
del U0W,V0S

# baroclinic velocity
ds1['maskL']=grid.interp(ds1.hFacC,'Z',to='left', boundary='extrapolate')  #mask for wvel
ds1['upW']=(ds1['UVEL']-ds1['U0W'])*ds1['maskW']
ds1['vpS']=(ds1['VVEL']-ds1['V0S'])*ds1['maskS']
wp=ds1['WVEL'].where(ds1['maskL'] !=0, np.nan)

# calculate horizontal dissipation
# setup other masks and hFac and dr
ds1['maskZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate') #for dvdx
ds1['hFacZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate') #for dvdx
ds1['hFacWL'] = grid.interp(ds1.hFacW, 'Z', to='left', boundary='extrapolate')  #for dwdx
print(ds1.hFacWL.dims)
ds1['hFacSL'] = grid.interp(ds1.hFacS, 'Z', to='left', boundary='extrapolate')  #for dwdy
print(ds1.hFacSL.dims)
ds1['drL']=grid.interp(ds1.drF, 'Z', to='left', boundary='extrapolate')

# 1 for dudx and dvdy, 2 for dvdx and dudy, 31 for dwdx, 32 for dwdy
# separately calculate because grids are different
hDispbc1 = rhoNil*viscAh*(((grid.diff(ds1.upW* ds1.dyG , 'X', boundary='extrapolate')/ds1.rA)**2\
                           #+(grid.diff(ds1.vpS* ds1.dxG , 'Y', boundary='extrapolate')/ds1.rA)**2
                           )*(ds1['drF']*ds1['hFacC'])).sum('Z')  #dudx & dvdy
hDispbc2 = rhoNil*viscAh*(((grid.diff(ds1.vpS* ds1.dyC , 'X', boundary='extrapolate')/ds1.rAz)**2\
                           #+(grid.diff(ds1.upW* ds1.dxC , 'Y', boundary='extrapolate')/ds1.rAz)**2
                           )*(ds1['drF']*ds1['hFacZ'])).sum('Z') #dvdx & dudy
dvdx=xr.DataArray(hDispbc2.values,coords=[ds1.time.values,yc,xc],dims=["time","YC","XC"])
print('dvdx')
print(dvdx)
hDispbc31 = rhoNil*viscAh*(((grid.diff(wp ,'X',boundary='extrapolate')/ds1.dxC)**2)*(ds1['drL']*ds1['hFacWL'])).sum('Zl')  #dwdx
#hDispbc32 = rhoNil*viscAh*(((grid.diff(wp ,'Y',boundary='extrapolate')/ds1.dyC)**2)*(ds1['drL']*ds1['hFacSL'])).sum('Zl')  #dwdy

# tidally averaged
ta_hDispbc1 = hDispbc1.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_hDispbc2 = dvdx.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
print('ta dvdx')
print(ta_hDispbc2)
ta_hDispbc31 = hDispbc31.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
#ta_hDispbc32 = hDispbc32.groupby_bins('time',time_bin,labels=time_bin_labels).mean()

ta_hDispbc = ta_hDispbc1+ta_hDispbc2+grid.interp(ta_hDispbc31,'X',boundary='extrapolate') 
#grid.interp(ta_hDispbc2,['X','Y'],boundary='extrapolate') \
                        #+grid.interp(ta_hDispbc32,'Y',boundary='extrapolate')
print('ta_hDispbc'+str(ta_hDispbc))


# dEbtdt, the rate of baroctropic energy
dtEbt=np.gradient(ds1.SDIAG1.values*rhoNil,dt,axis=0)
print('dtEbc'+str(dtEbt.shape))
ddtEbt = xr.DataArray(dtEbt.data, coords=[ds1.time.values,yc,xc], dims=['time','YC', 'XC'])
ta_dtEbt = ddtEbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()

# dEbcdt, the rate of baroclinic energy
Ebc=rhoNil*ds1['SDIAG5']
dtEbc=np.gradient(Ebc,dt,axis=0)
#print('dtEbc'+str(dtEbc.shape))
ddtEbc = xr.DataArray(dtEbc.data, coords=[ds1.time.values,yc,xc], dims=['time','YC', 'XC'])
ta_dtEbc = ddtEbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
del Ebc, dtEbc
print('ta_dtEbc'+str(ta_dtEbc.shape))

# hdFbt
Fxbt=ds2.SDIAG2+ds2.SDIAG4
Fybt=ds2.SDIAG3
Fxbt=xr.DataArray(rhoNil*Fxbt.values, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
Fybt=xr.DataArray(rhoNil*Fybt.values, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
ta_Fxbt=Fxbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fybt=Fybt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
hd_ta_Fbt=(grid.diff(ta_Fxbt*ds2['dyG'],'X',boundary='extrapolate'))/ds2['rA'] 
#+grid.diff(ta_Fybt*ds2['dxG'],'Y',boundary='extrapolate')

# hdFbc
uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
#print(uPbc)
#print(uEbc)
Fxbc=uPbc+uEbc
Fybc=vPbc+vEbc
ta_Fxbc=Fxbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fybc=Fybc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
#print(Fbc)
hd_ta_Fbc=(grid.diff(ta_Fxbc*ds2['dyG'],'X',boundary='extrapolate'))/ds2['rA'] 
#+grid.diff(ta_Fybc*ds2['dxG'],'Y',boundary='extrapolate')
del uPbc, vPbc, uEbc, vEbc, Fxbc, Fybc
print('hd_ta_Fbc'+str(hd_ta_Fbc))

# BC-BT conversion
Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[ds2.time.values,yc,xc], dims=['time','YC','XC'])
#print('Conv'+str(Conv))
ta_Conv=Conv.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
print('ta_Conv'+str(ta_Conv))

# kl10 vertical dissipation
dz=200/40 #H/nz
Dsp=np.sum(rhoNil*ds1['KLeps']*dz,axis=1)
print(Dsp)
vDsp=xr.DataArray(Dsp.data, coords=[ds1.time.values,yc,xc], dims=['time','YC','XC'])
#print('vDsp'+str(vDsp))
ta_dsp=vDsp.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
print('ta_dsp'+str(ta_dsp))

# bottom drag
Cd=1e-3

wall_id=ds1.Depth.where(ds1.Depth!=0,111111) #wall as 111111
wall_id=wall_id.where(wall_id==111111,0) #nonwall as 0
wall_id=wall_id.where(wall_id==0,1) #wall as 1
wall_id=xr.DataArray(wall_id.values,coords=[yc,xc],dims=("YC", "XC")) #turn dask to xarray
print(wall_id)

bottom_ind = xr.DataArray(np.floor(ds1.hFacC.sum('Z')).values.astype(int),coords=[yc,xc],dims=("YC", "XC"))
print(bottom_ind)
bottom_ind = bottom_ind.where(bottom_ind<40,39) # for depth=200, sum(hFac)=40 but inds should be 39
print(bottom_ind)
bottom_ind = bottom_ind*(-1*(wall_id-1)).astype(int)  #(-1*(wall_id-1)) is a nonwall mask
print(bottom_ind)

bottom_ind.plot()
plt.savefig('./figs/bottomid.png')
print('bottom_ind.chunks:')
print(bottom_ind.chunks)

UC=grid.interp(ds1.UVEL,'X', boundary='extrapolate')
#VC=grid.interp(ds1.VVEL,'Y', boundary='extrapolate')
U0C=((UC*ds1['drF']*ds1['hFacC']).sum('Z'))/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')
#V0C=((VC*ds1['drF']*ds1['hFacC']).sum('Z'))/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')
wC = grid.interp(ds1.WVEL,'Z', boundary='extrapolate')
Ubot=UC.isel(Z=bottom_ind)
#Vbot=VC.isel(Z=bottom_ind)
wbot=wC.isel(Z=bottom_ind)

upCbot=(Ubot-U0C)  #baroclinic bottom velocity
#vpCbot=(Vbot-V0C)
absuH=(Ubot**2)**0.5#+Vbot**2
D0=rhoNil*Cd*absuH*(Ubot*U0C)#+Vbot*V0C)
Dp=rhoNil*Cd*absuH*(Ubot*upCbot+wbot*wbot) #Vbot*vpCbot
print('Dp:')
print(Dp)
print('D0:')
print(D0)

ta_D0=D0.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Dp=Dp.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
print('ta_D0'+str(ta_D0))

del dtEbt, ddtEbt, Fxbt, Fybt, Conv,  U0C, D0, Dp #VC,V0C

ta_Energy=xr.Dataset({"dEbt/dt": ta_dtEbt,"dEbc/dt": ta_dtEbc,"divFbc": hd_ta_Fbc,"divFbt": hd_ta_Fbt,"BC-BT Conv": ta_Conv,"vDissp": ta_dsp,"hDissp": ta_hDispbc,'BD0':ta_D0,'BDp':ta_Dp})

ta_Energy.to_netcdf('ta_Energy.nc')

#df = pd.DataFrame({"dEbt/dt": ta_dtEbt.values,"dEbc/dt": ta_dtEbc.values,"divFbc": hd_ta_Fbc.values,"divFbt": hd_ta_Fbt.values,"BC-BT Conv": ta_Conv.values,"vDissp": ta_dsp.values,"hDissp": ta_hDispbc.values,'BD0':ta_D0.values,'BDp':ta_Dp.values})
#df.to_csv("TidallyAveragedEnergyBudget.csv")

#read_df = pd.read_csv("TidallyAveragedEnergyBudget.csv")
#print(read_df)












