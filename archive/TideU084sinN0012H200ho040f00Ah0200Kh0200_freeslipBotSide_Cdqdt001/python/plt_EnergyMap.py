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


if 1:
    #t=20
    f, ax = plt.subplots(figsize=(10,12))
    #print(ds['UVEL'].isel(time=0,YC=0))
    Ebc=rhoNil*ds2['SDIAG5']
    print('Ebc' + str(Ebc))
    uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'], coords=[time,yg,xc], dims=['time','YG','XC'])
    uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'], coords=[time,yg,xc], dims=['time','YG','XC'])
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[time,yc,xc], dims=['time','YC','XC'])
    dissp=ds1['KLeps'].isel(time=range(len(time)-1))
    Dsp=-dissp.integrate("Zl")
    print('Dsp' + str(Dsp))
    dmax=2#np.amax(Dsp.values)
    dmin=10**(-10)
    #print(uPbc)
    #print(uEbc)
    Fxbc=uPbc+uEbc
    Fybc=vPbc+vEbc
    print('Fxbc' + str(Fxbc))
    print('Fybc' + str(Fybc))
    hdFbc=(grid.diff(Fxbc*ds2['dyG'],'X', boundary='extrapolate')+grid.diff(Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']#*ds['rA']
    print('hdFbc' + str(hdFbc))
    dt=1860

    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time,yc,xc], dims=['time','YC', 'XC'])
    Res=Conv-hdFbc-ddtEbc
    print('Res' + str(Res))
   
    if 0:
        ddtEbc.plot(x='XC',cmap='RdBu_r',y='time',col='YC',col_wrap=4,vmin=-30,vmax=30,cbar_kwargs={"label": "", "aspect": 40})
        #Ebc.plot(ax=ax[0],cmap='RdBu_r',y='time',cbar_kwargs={"label": "", "aspect": 40})
        #ax[0].set_title('depth-integrated Ebc')
        #ax[0].set_ylabel("time [hr]")
        #ax[0].set_xlabel("X [km]")
        #plt.show()
        plt.savefig('./figs/dEdt_xt.png')
        plt.clf()

    if 1:
        hdFbc.plot(x='XC',y='time',col='YC',col_wrap=4,cmap='RdBu_r',vmin=-200,vmax=200,cbar_kwargs={"label": "", "aspect": 40})
        #ax[1].set_title('depth-integrated hd(Fbc)')
        #ax[1].set_ylabel("time [hr]")
        #ax[1].set_xlabel("X [km]")
        plt.savefig('./figs/hdFbc_xt.png')
        plt.clf()

    if 0:
        Conv.plot(x='XC',y='time',col='YC',col_wrap=4,cmap='RdBu_r',vmin=-100,vmax=100,cbar_kwargs={"label": "", "aspect": 40})
        #ax[2].set_title('depth-integrated Conv')
        #ax[2].set_ylabel("time [hr]")
        #ax[2].set_xlabel("X [km]")
        plt.savefig('./figs/Conv_xt.png')
        plt.clf()
    
    if 1:
        Res.plot(x='XC',y='time',col='YC',col_wrap=4,cmap='RdBu_r',vmin=-200,vmax=200,cbar_kwargs={"label": "", "aspect": 40})
        #ax[3].set_title('Residual= Conv-hdFbc-dtEbc')
        #ax[3].set_ylabel("time [hr]")
        #ax[3].set_xlabel("X [km]")
        plt.savefig('./figs/Res_xt.png')
        plt.clf()

    if 0:
        Dsp.plot(x='XC',y='time',col='YC',col_wrap=4,cmap='RdBu_r',vmin=dmin, vmax=dmax, norm=colors.LogNorm(vmin=dmin, vmax=dmax), cbar_kwargs={"label": "", "aspect": 40})
        #ax[4].set_title('depth-integrated dissipation')
        #ax[4].set_ylabel("time [hr]")
        #ax[4].set_xlabel("X [km]")
        plt.savefig('./figs/Dsp_xt.png')
        plt.clf()



