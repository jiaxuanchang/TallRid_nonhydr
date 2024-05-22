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
#h0 = 340.
#H = float(currentDirectory[-15:-12])
#N0 = float(currentDirectory[-20:-16])/1e6
#Fr=int(currentDirectory[-26:-22])/1e4
#U0=Fr*N0*h0
f0 = 1.e-4
g = 9.8

om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
#dz=H/nz 
Tref=23#35+np.cumsum(N0**2/g/alpha*(-H))

xmin = 40
xmax = 44
numcolt=21
numcolv=21
                                        
ttlen=len(ds.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(Tref))


if 1:
    Ebc=ds['SDIAG5'].isel(YC=0)
    dt=1860

    #ds.coords['time']=ds.coords["time"]/3600
    ds.coords['XG'] = ds.coords['XG']/1000
    ds.coords['XC'] = ds.coords['XC']/1000
    time=ds.coords['time']
    xc=ds.coords['XC']
    
    print(np.shape(Ebc.values))
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time,xc], dims=['time', 'XC'])
    plt.plot(time,Ebc.isel(XC=180).values)
    plt.plot(time,1000*dtEbc[:,180])
    plt.show()

    #plt.savefig('./figs/Energy_xt.png')
    #plt.tight_layout()
    #plt.show()


