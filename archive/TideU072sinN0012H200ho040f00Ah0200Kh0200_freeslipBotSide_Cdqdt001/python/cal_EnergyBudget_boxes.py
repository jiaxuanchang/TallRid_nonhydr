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
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(tRef))

time1=ds1.coords['time']
time=ds2.coords['time']
xc=ds1.coords['XC']
xg=ds2.coords['XG']
yc=ds2.coords['YC']
yg=ds2.coords['YG']
yc0 = (ds1.YC/1000.-ds1.YC.mean()/1000.)
z=ds1.coords['Z']

ys=range(0,3200,600)

ix=[i for i, e in enumerate(xc) if (e > xmin) & (e < xmax)]
print(ix)
print(xc[ix[-1]])
print(xg[335:338])
ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])

if 1:
    f = plt.figure(figsize=(15,10))
    gs = plt.GridSpec(3, 2)

    rho=(rhoNil*(1-alpha*(ds1['THETA']-ds1['tRef'])+beta*(ds1['SALT']-refSalt)))*ds1['maskC']
    print(rho)
    rhol=grid.interp(rho,'Z',boundary='extrapolate')
    dt=time.values[1]-time.values[0]
    print(dt)
    
    # dEdt
    Ebc=rhoNil*ds2['SDIAG5']
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[time,yc,xc], dims=['time','YC', 'XC'])
    
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

    # conversion BC-BT
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[time,yc,xc], dims=['time','YC','XC'])

    # dissipation from kl10
    dissp=ds1['KLeps'].isel(time=range(len(time)))
    Dsp=-(rhol*dissp).integrate("Zl")
    dmax=2#np.amax(Dsp.values)
    dmin=10**(-10)
    
    for ii in range(0,5):
        iy=[i for i, e in enumerate(yc) if (e > ys[ii]) & (e < ys[ii+1])]
        # SUMMATION
        DtdE= (ddtEbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                    ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        print('dtde' + str(DtdE))
        BCradx=(hdFxbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                     ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        BCrady=(hdFybc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                     ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        BCrad=BCradx+BCrady
        #print(hdFbc)
        fxdy=((Fxbc*ds2['dyG']).isel(XG=ix[0])-(Fxbc*ds2['dyG']).isel(XG=ix[-1]+1)).sum('YC')/1e6
        fydx=-((Fybc*ds2['dxG']).isel(YG=iy[0])-(Fybc*ds2['dxG']).isel(YG=iy[-1])).sum('XC')/1e6
        BCrad1=fxdy+fydx
        C=(Conv*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                              ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        R1=((Conv-ddtEbc)*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                       ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6-BCrad1
        R2=((Conv-ddtEbc-hdFxbc-hdFybc)*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                                      ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        D=(Dsp*ds1['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                             ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6

        
        # PLOT
        ax = f.add_subplot(gs[ii])
        
        DtdE.plot(ax=ax,color='tab:blue',label="dEdt")
        #BCradx.plot(ax=ax,color='tan',ls=':',label=r"$\sum (\frac{\partial F_x^{\prime}}{\partial x})dA$")
        #BCrady.plot(ax=ax,color='goldenrod',ls=':',label=r"$\sum (\frac{\partial F_y^{\prime}}{\partial y})dA$")
        BCrad.plot(ax=ax,color='tab:orange',label=r"$\sum (\frac{\partial F_x^{\prime}}{\partial x}+\frac{\partial F_y^{\prime}}{\partial y})dA$")
        #fxdy.plot(ax=ax,color='plum',ls=':',label="$\sum F_x^{\prime}dy$")
        #fydx.plot(ax=ax,color='palevioletred',ls=':',label="$\sum F_y^{\prime}dx$")
        #BCrad1.plot(ax=ax,color='tab:purple',label="$\sum F_x^{\prime}dy+\sum F_y^{\prime}dx$")
        C.plot(ax=ax,color='tab:green',label="BT-BC Conversion")
        #R1.plot(color='grey',ax=ax,label=r"Res=$\sum (C-\frac{\partial}{\partial t}E)dA-\sum F_x dy$")
        R2.plot(color='slategrey',ax=ax,label=r"Res=$\sum (C-\frac{\partial}{\partial t}E-\nabla F^{\prime})dA dy$")
        D.plot(ax=ax,color='tab:red',label="Dissipation")
        #plt.legend(('dEdt',r'$\nabla  F^{\prime}$','BT-BC Conversion',r'Res=$ C-\frac{\partial}{\partial t}E-\nabla F^{\prime}$','Dissipation'))
        ax.set_ylim(-2,11)
        if ii==4:
            ax.set_ylabel('Energy Budget [MW]')
            ax.set_xlabel('time [s]')
            ax.set_title('YC= %1.1f ~ %1.1f km ' %(yc0[iy[0]], yc0[iy[-1]]))
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.set_title('YC= %1.1f ~ %1.1f km ' %(yc0[iy[0]], yc0[iy[-1]]))
    handles, labels = ax.get_legend_handles_labels()   
    f.legend(labels,loc='lower right',bbox_to_anchor=(0.93, 0.08),ncol=3 )


    plt.tight_layout()
    plt.savefig('./figs/EnergyBudget_x34_50_5box_t.png')
    #plt.show()


