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


# hdFbt
uPbt=xr.DataArray(rhoNil*ds2.SDIAG2.data,coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vPbt=xr.DataArray(rhoNil*ds2.SDIAG3.data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
uEbt=xr.DataArray(rhoNil*ds2.SDIAG4.data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
Fxbt=uPbt+uEbt
Fybt=vPbt #there is no vEbt output from diag

ta_uPbt=uPbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_vPbt=vPbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_uEbt=uEbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fxbt=Fxbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fybt=Fybt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()

# hdFbc
uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'].data, coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'].data, coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
#print(uPbc)
#print(uEbc)
Fxbc=uPbc+uEbc
Fybc=vPbc+vEbc
ta_uPbc=uPbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_uEbc=uEbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_vPbc=vPbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_vEbc=vEbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fxbc=Fxbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
ta_Fybc=Fybc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
#print(Fbc)


ta_Flux=xr.Dataset({"ta_uPbc": ta_uPbc,"ta_uEbc": ta_uEbc,"ta_vPbc": ta_vPbc,"ta_vEbc": ta_vEbc,"ta_Fxbc": ta_Fxbc,"ta_Fybc": ta_Fybc,"ta_uPbt": ta_uPbt,"ta_uEbt": ta_uEbt,"ta_vPbt": ta_vPbt,"ta_Fxbt": ta_Fxbt,"ta_Fybt": ta_Fybt})

ta_Flux.to_netcdf('ta_Flux.nc')

#df = pd.DataFrame({"dEbt/dt": ta_dtEbt.values,"dEbc/dt": ta_dtEbc.values,"divFbc": hd_ta_Fbc.values,"divFbt": hd_ta_Fbt.values,"BC-BT Conv": ta_Conv.values,"vDissp": ta_dsp.values,"hDissp": ta_hDispbc.values,'BD0':ta_D0.values,'BDp':ta_Dp.values})
#df.to_csv("TidallyAveragedEnergyBudget.csv")

#read_df = pd.read_csv("TidallyAveragedEnergyBudget.csv")
#print(read_df)












