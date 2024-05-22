from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
import numpy as np
import math
from scipy import integrate
import xarray as xr
import string

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
grid = xgcm.Grid(ds1, periodic=False)
print(grid)

t = 0
#h0 = 340.
#H = float(currentDirectory[-15:-12])
#N0 = float(currentDirectory[-20:-16])/1e6
#Fr=int(currentDirectory[-26:-22])/1e4
#U0=Fr*N0*h0
f0 = 1.e-4
g = 9.8
rhoNil=999.8

om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
#dz=H/nz 
Tref=23#35+np.cumsum(N0**2/g/alpha*(-H))

xmin = 40
xmax = 44
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(Tref))

time=ds2.coords['time']
xc=ds2.coords['XC']
xg=ds2.coords['XG']
yc=ds2.coords['YC']
yg=ds2.coords['YG']
#print(xg)
#print(yg)

ds1=ds1.assign_coords(XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.))
ds1=ds1.assign_coords(XG0 = (ds1.XG-ds1.XC.mean())/1000.)
ds1=ds1.assign_coords(YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.))
ds1=ds1.assign_coords(YG0 = (ds1.YG-ds1.YC.mean())/1000.)

xge=(ds1['XG0'][-1]+ds1['XC0'][-1]-ds1['XC0'][-2]).values
yge=(ds1['YG0'][-1]+ds1['YC0'][-1]-ds1['YC0'][-2]).values
xg0=np.array(ds1['XG0'].values)
xg0=np.hstack((xg0,xge))
yg0=ds1['YG0'].values
yg0=np.hstack((yg0,yge))


if 1:
    #t=20
    f, ax = plt.subplots(1,2,figsize=(10,5))
    #print(ds['UVEL'].isel(time=0,YC=0))
    Dep=ds1['Depth']
    print(Dep)

    if 1:
        ax[0].pcolormesh(xg0,yg0,Dep,cmap='YlGnBu')
        ax[0].set_ylim([-1.5,1.5])
        ax[0].set_xlabel('X [km]')
        ax[0].set_ylabel('Y [km]')

        cf=ax[1].pcolormesh(xg0,yg0,Dep,cmap='YlGnBu')
        ax[1].set_ylim([-1.5,1.5])
        ax[1].set_xlim([-2.,2.])
        ax[1].set_xlabel('X [km]')
        cb=f.colorbar(cf,ax=ax[1])
        cb.ax.set_ylabel('Ocean Depth [m]')

        plt.savefig('./figs/DepthMap_YlGnBu.png')

