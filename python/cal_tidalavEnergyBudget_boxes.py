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
yc0 = (ds1.YC/1000.-ds1.YC.mean()/1000.)
z=ds1.coords['Z']
zl=ds1.coords['Zl']

ys=range(0,3200,600)

ix=[i for i, e in enumerate(xc) if (e > xmin) & (e < xmax)]
print(ix)

ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])

if 1:
    #t=20
    #f, ax = plt.subplots(1, 5, figsize=(20,9) , sharey=True)
    #print(ds1['UVEL'].isel(time=0,YC=0))
    f=plt.figure(figsize=(15,10))
    gs = plt.GridSpec(3, 2)
    #host=host_subplot(111,figure=f,axes_class=AA.Axes)
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
    ta_dtEbc = ddtEbc.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60),labels=time_bin_labels).mean()
    
    # hdFbc
    uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'], coords=[time,yg,xc], dims=['time','YG','XC'])
    uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'], coords=[time,yc,xg], dims=['time','YC','XG'])
    vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'], coords=[time,yg,xc], dims=['time','YG','XC'])
    #print(uPbc)
    #print(uEbc)
    Fxbc=uPbc+uEbc
    Fybc=vPbc+vEbc
    ta_Fxbc=Fxbc.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60),labels=time_bin_labels).mean()
    ta_Fybc=Fybc.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60),labels=time_bin_labels).mean()
    #print(Fbc)
    hd_ta_Fbc=(grid.diff(ta_Fxbc*ds2['dyG'],'X',boundary='extrapolate')+grid.diff(ta_Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']

    # BC-BT conversion
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[time,yc,xc], dims=['time','YC','XC'])
    print('Conv'+str(Conv))
    ta_Conv=Conv.groupby_bins('time',np.arange(0.,time.values[-1],12.4*60*60),labels=time_bin_labels).mean()
    print('ta_Conv'+str(ta_Conv))
    
    # dissipation
    Dsp=-(rhoNil*ds1['KLeps']).integrate("Zl")
    print('Dsp'+str(Dsp))
    ta_dsp=Dsp.groupby_bins('time',np.arange(0.,time1.values[-1],12.4*60*60),labels=time_bin_labels).mean()
    print('ta_dsp'+str(ta_dsp))
    dmax=2#np.amax(Dsp.values)
    dmin=10**(-10)
 
    for ii in range(0,5):

        iy=[i for i, e in enumerate(yc) if (e > ys[ii]) & (e < ys[ii+1])]
        # SUMMATION
        DtdE= (ta_dtEbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                      ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
    
        BCrad=(hd_ta_Fbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                       ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        #fxdy=((ta_Fxbc*ds2['dyG']).isel(XG=ix[-1])-(ta_Fxbc*ds2['dyG']).isel(XG=ix[0])).sum('YC')/1e6
        C=(ta_Conv*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                 ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        print('C'+str(C))
        R=((ta_Conv-ta_dtEbc-hd_ta_Fbc)*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                     ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        print(R)
        D=(ta_dsp*ds1['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)]
                                ,YC=yc[(yc > ys[ii]) & (yc < ys[ii+1])]).sum(['XC','YC'])/1e6
        print(D)
        
        RAT1=D/C*100
        RAT2=R/C*100
        print('rat1' +str(RAT1))

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
        R.plot(color='slategrey',ax=ax,label=r"Res=$\sum (C-\frac{\partial}{\partial t}E-\nabla F^{\prime})dA dy$")
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

        if 0:
            ax2=host.twinx()
            par2=host.twinx()
            
            offset = 0
            new_fixed_axis = par2.get_grid_helper().new_fixed_axis
            par2.axis["right"] = new_fixed_axis(loc="right",axes=par2,offset=(offset, 0))
            par2.axis["right"].toggle(all=True)

            host.set_ylim(-0.2,10)
            host.set_xlim(0,400000)
            

            host.set_ylabel("Energy Budget [MW]")
            host.set_xlabel("time [s]")
            par2.set_ylabel("Percentage [%]") 
         
            p1, = host.plot(DtdE.time_bins,DtdE.values,'o-',label="dE/dt")
            #BCrad.plot()
            host.plot(fxdy.time_bins,fxdy,'o-',label=r"$\nabla F_{bc}$")
            host.plot(C.time_bins,C,'o-',label="BT-BC Conversion")
            host.plot(R.time_bins,R,'o-',color='grey',label=r"Res=Conv-dE/dt-$\nabla F_{bc}$")
            host.plot(D.time_bins,D,'o-',label=r"Dissipation=$\int \rho \varepsilon dV$")
            host.plot(time1,np.zeros(ttlen),'k--')
            
            p2, = ax2.plot(RAT1.time_bins,RAT1,label="Disp/Conv")
            p3,= par2.plot(RAT2.time_bins,RAT2,label="Res/Conv")
            ax2.set_ylim(-5,100)
            par2.set_ylim(-5,100)

            host.legend()
            #plt.legend(('dE/dt',r'$\nabla F_{bc}$','BT-BC Conversion',r'Res=Conv-dE/dt-$\nabla F_{bc}$',r'Dissipation=$\int \rho \varepsilon dV$'))
        
    

        plt.tight_layout()
        plt.savefig('./figs/EnergyBudget_t_tidal_5box.png')
    #plt.show()


