from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import pandas as pd
import numpy as np
import math
from scipy import integrate
import xarray as xr
import string
import dask.array as da

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])

grid = xgcm.Grid(ds1, periodic=False)
print(grid)

t = 0
f0 = 1.e-4
g = 9.8
rhoNil=999.8

om=2*np.pi/12.4/3600
alpha = 2e-4
beta = 0.
nz = 200
#dz=H/nz 
tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.

xmin = 34000
xmax = 50000
ymin = 0000
ymax = 3000
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(tRef))

#ds1.coords['time']=ds1.coords["time"]/3600
#ds1.coords['XG'] = ds1.coords['XG']/1000
#ds1.coords['XC'] = ds1.coords['XC']/1000
time1=ds1.coords['time']
time=ds2.coords['time']
xc=ds1.coords['XC']
xg=ds2.coords['XG']
yc=ds2.coords['YC']
yg=ds2.coords['YG']
z=ds1.coords['Z']
zl=ds1.coords['Zl']
ix=[i for i, e in enumerate(xc) if (e > xmin) & (e < xmax)]
iy=[i for i, e in enumerate(yc) if (e > ymin) & (e < ymax)]
print(ix)
print(iy)

ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])

plt.clf()
if 1:
    #t=20
    #f, ax = plt.subplots(1, 5, figsize=(20,9) , sharey=True)
    #print(ds1['UVEL'].isel(time=0,YC=0))
    f=plt.figure(figsize=(15,6))
    host=host_subplot(111,figure=f,axes_class=AA.Axes)
    rho=(rhoNil*(1-alpha*(ds1['THETA']-ds1['tRef'])+beta*(ds1['SALT']-refSalt)))*ds1['maskC']
    print(rho)
    rhol=grid.interp(rho,'Z',boundary='extrapolate')
    #THIS rhol CHUNK SIZE DECREASE BY ONE, ?
    #print(hdFbc)
    dt=time.values[1]-time.values[0]
    print(dt)

    time_bin_labels = np.arange(12.4*60*60/2,time.values[-1]-20000,12.4*60*60)
    print(time_bin_labels)
    
    # dEdt
    Ebc=rhoNil*ds2['SDIAG5']
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time,yc,xc], dims=['time','YC', 'XC'])
    ta_dtEbc = ddtEbc.groupby_bins('time'
                                   ,np.arange(0.,time.values[-1],12.4*60*60)
                                   ,labels=time_bin_labels).mean()
    print(ta_dtEbc)

    # hdFbc
    uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'], coords=[time,yg,xc], dims=['time','YG','XC'])
    uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'], coords=[time,yg,xc], dims=['time','YG','XC'])
    #print(uPbc)
    #print(uEbc)
    Fxbc=uPbc+uEbc
    Fybc=vPbc+vEbc
    ta_Fxbc=Fxbc.groupby_bins('time'
                              ,np.arange(0.,time.values[-1],12.4*60*60)
                              ,labels=time_bin_labels).mean()
    ta_Fybc=Fybc.groupby_bins('time'
                              ,np.arange(0.,time.values[-1],12.4*60*60)
                              ,labels=time_bin_labels).mean()
    #print(Fbc)
    hd_ta_Fbc=(grid.diff(ta_Fxbc*ds2['dyG'],'X',boundary='extrapolate')
              +grid.diff(ta_Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']

    # BC-BT conversion
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[time,yc,xc], dims=['time','YC','XC'])
    print('Conv'+str(Conv))
    ta_Conv=Conv.groupby_bins('time'
                              ,np.arange(0.,time.values[-1],12.4*60*60)
                              ,labels=time_bin_labels).mean()
    
    R=ta_Conv-ta_dtEbc-hd_ta_Fbc
    print(R)

    # dissipation
    Dsp=-(rhoNil*ds1['KLeps']).integrate("Zl")
    print('Dsp'+str(Dsp))
    ta_dsp=Dsp.groupby_bins('time'
                            ,np.arange(0.,time1.values[-1],12.4*60*60)
                            ,labels=time_bin_labels).mean()
    print('ta_dsp'+str(ta_dsp))
    
    Dep=ds1['Depth']
    levels=range(0,210,50)
    # plotting
    T=7
    fig, axes = plt.subplots(figsize=(15,9.5), nrows=3, ncols=2, sharex='all', sharey='all')
    axes=axes.flatten()
    fig.subplots_adjust(top=0.9,left=0.08,right=0.95)
    fig.suptitle("t= %1d tidal" %T, size=16)
    
    ta_dtEbc.isel(time_bins=T).plot(ax=axes[0],cmap='RdBu_r')
    Dep.plot.contour(ax=axes[0],levels=levels,colors='grey',linewidths=0.7)
    axes[0].set_title(r'$\langle \overline{\frac{\partial E}{\partial t}}\rangle$')
    axes[0].set_xlim(34000,50000)
    
    hd_ta_Fbc.isel(time_bins=T).plot(ax=axes[1],cmap='RdBu_r')
    Dep.plot.contour(ax=axes[1],levels=levels,colors='grey',linewidths=0.7)
    axes[1].set_title(r'$\nabla_{H} \cdot \langle \overline{\mathbf{F}^{\prime}} \rangle$')
    axes[1].set_xlim(34000,50000)
    
    ta_Conv.isel(time_bins=T).plot(ax=axes[2],cmap='RdBu_r')
    Dep.plot.contour(ax=axes[2],levels=levels,colors='grey',linewidths=0.7)
    axes[2].set_title(r'$\langle \overline{\mathbf{C}}  \rangle$')
    axes[2].set_xlim(34000,50000)
    
    R.isel(time_bins=T).plot(ax=axes[3],cmap='RdBu_r')
    Dep.plot.contour(ax=axes[3],levels=levels,colors='grey',linewidths=0.7)
    axes[3].set_title(r'$\langle\bar{C}\rangle - \langle \overline{\frac{\partial E}{\partial t}}\rangle  -\nabla_{H} \cdot\langle\overline{\mathbf{F}^{\prime}}\rangle$')
    axes[3].set_xlim(34000,50000)
    
    ta_dsp.isel(time_bins=T).plot(ax=axes[4],cmap='RdBu_r')
    Dep.plot.contour(ax=axes[4],levels=levels,colors='grey',linewidths=0.7)
    axes[4].set_title(r'$\langle\bar{\varepsilon}\rangle$')
    axes[4].set_xlim(34000,50000)
    
    axes[5].set_visible(False)
    plt.savefig('./figs/EnergyBudget_xy_tidalP7.png')
    #plt.show()


