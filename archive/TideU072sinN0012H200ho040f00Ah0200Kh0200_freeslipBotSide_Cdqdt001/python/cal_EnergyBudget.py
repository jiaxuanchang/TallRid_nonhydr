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
ymin = 0
ymax = 10000
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
ix=[i for i, e in enumerate(xc) if (e > xmin) & (e < xmax)]
iy=[i for i, e in enumerate(yc) if (e > ymin) & (e < ymax)]
print(ix)
print(iy)
ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])

plt.clf()
if 1:
    plt.figure(figsize=(15,6))
    rho=(rhoNil*(1-alpha*(ds1['THETA']-ds1['tRef'])+beta*(ds1['SALT']-refSalt)))*ds1['maskC']
    print(rho)
    rhol=grid.interp(rho,'Z',boundary='extrapolate')
    dt=time.values[1]-time.values[0]
    print(dt)
    
    # dEdt
    Ebc=rhoNil*ds2['SDIAG5']
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time,yc,xc], dims=['time','YC', 'XC'])
    DtdE= (ddtEbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print(DtdE)
    
    # div(F')
    uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'], coords=[time,yg,xc], dims=['time','YG','XC'])
    uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'], coords=[time,yg,xc], dims=['time','YG','XC'])
    #print(uPbc)
    #print(uEbc)
    Fxbc=uPbc+uEbc
    Fybc=vPbc+vEbc
    #print(Fbc)
    hdFxbc=(grid.diff(Fxbc*ds2['dyG'],'X',boundary='extrapolate'))/ds2['rA']
    hdFybc=(grid.diff(Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']
    BCradx=(hdFxbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    BCrady=(hdFybc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    BCrad=BCradx+BCrady
    #print(hdFbc)
    fxdyl=(Fxbc*ds2['dyG']).isel(XG=ix[0]).sum('YC')
    fxdyr=(Fxbc*ds2['dyG']).isel(XG=ix[-1]).sum('YC')
    print(fxdyl)
    print(fxdyr)
    fxdy=(fxdyr-fxdyl)/1e6
    print(fxdy.values)

    # conversion BC-BT
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[time,yc,xc], dims=['time','YC','XC'])
    C=(Conv*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print(C)

    # residual
    R=((Conv-ddtEbc)*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6-fxdy
    print('R'+ str(R))
    
    # dissipation from kl10
    dissp=ds1['KLeps'].isel(time=range(len(time)))
    Dsp=-(rhol*dissp).integrate("Zl")
    dmax=2#np.amax(Dsp.values)
    dmin=10**(-10)
    D=(Dsp*ds1['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print(D)

    DtdE.plot()
    #BCradx.plot()
    #BCrady.plot()
    #BCrad.plot()
    fxdy.plot()
    C.plot()
    R.plot(color='grey')
    D.plot()
    plt.plot(time1,np.zeros(ttlen),'k--')
    #plt.legend(('dEdt',r'$\sum (\frac{\partial F_x^{\prime}}{\partial x})dA$',r'$\sum (\frac{\partial F_y^{\prime}}{\partial y})dA$',r'$\sum (\frac{\partial F_x^{\prime}}{\partial x}+\frac{\partial F_y^{\prime}}{\partial y})dA$','$\sum F_x^{\prime}dy$','BT-BC Conversion',r'Res=$\sum (C-\frac{\partial}{\partial t}E)dA-\sum F_x dy$','Dissipation'))
    plt.legend(('dEdt',r'$\nabla  F^{\prime}$','BT-BC Conversion',r'Res=$ C-\frac{\partial}{\partial t}E-\nabla F^{\prime}$','Dissipation'))
    plt.ylabel('Energy Budget [MW]')
    plt.xlabel('time [s]')

    #plt.show()

    plt.tight_layout()
    plt.savefig('./figs/EnergyBudget_x34_50_y0_3_t.png')
    #plt.show()


