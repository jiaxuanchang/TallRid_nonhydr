from xmitgcm import open_mdsdataset
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
import numpy as np
import math
from scipy import integrate
import xarray as xr

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds = open_mdsdataset(data_dir, geometry='cartesian', endian='<')

t = 0
f0 = 1.e-4
g = 9.8

om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
Tref=23#35+np.cumsum(N0**2/g/alpha*(-H))

xmin = 34000
xmax = 50000
numcolt=21
numcolv=21
                                        
ttlen=len(ds.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(Tref))

#ds.coords['time']=ds.coords["time"]/3600
#ds.coords['XG'] = ds.coords['XG']/1000
#ds.coords['XC'] = ds.coords['XC']/1000
time=ds.coords['time']
xc=ds.coords['XC']
dt=time.values[1]-time.values[0]
#print(ds)

f, axes = plt.subplots(2, 3, figsize=(12,8) ,sharex=True, sharey=True)
#plt.figure(figsize=(15,6))
#plt.clf()
axes=axes.flatten()
xind=range(190,220,5)
print(xind)

for ax, i in zip(axes, xind): 
    
    print(i)
    Ebc=ds['SDIAG5'].isel(XC=i,YC=0)
    uPbc=ds['SDIAG6'].isel(YC=0)
    uEbc=ds['SDIAG8'].isel(YG=0)
    Conv=ds['SDIAG10'].isel(XC=i,YG=0)
    dissp=ds['KLeps'].isel(time=range(0,244,1),XC=i,YC=0)
    Dsp=-dissp.integrate("Zl")
    
    ta_Conv=Conv.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60)).mean()
    ta_dsp=Dsp.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60)).mean()

    Fbc=uPbc+uEbc
    ta_Fbc=Fbc.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60)).mean()
    hd_ta_Fbc=ta_Fbc.differentiate("XC").isel(XC=i)
   
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time], dims=['time'])
    ta_dtEbc = ddtEbc.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60)).mean()

    Res=ta_Conv-hd_ta_Fbc-ta_dtEbc
 
    print(hd_ta_Fbc)
    print(ta_dtEbc)
    print(Res)

    ta_dtEbc.plot(ax=ax)
    hd_ta_Fbc.plot(ax=ax)
    ta_Conv.plot(ax=ax)
    Res.plot(ax=ax,color='grey')
    ta_dsp.plot(ax=ax)
    ax.plot(time,np.zeros(ttlen),'k--')
    ax.set_ylabel('Tidally-av Energy Budget [$W m^{-2}$]')
    ax.set_xlabel('time [s]')
    ax.set_title('XC: %i m' %xc[i]  )
    #ax.set_ylim((-0.2,0.2))

axes[4].legend(('dEdt','BC flux divergence','BT-BC Conversion','Residual','Dissipation'),loc='upper center',bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=5)

#plt.show()
plt.savefig('./figs/tidalav_EnergyBudget_t_1grid.png')
#plt.tight_layout()
    #plt.show()


