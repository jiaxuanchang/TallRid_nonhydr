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
import pandas as pd

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])

grid = xgcm.Grid(ds1, periodic=False)
print(grid)
ds1['hFacZ'] = grid.interp(ds1.hFacW,'Y',boundary='extrapolate')

t = 0
viscAh=1.e-4
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

xmin = 15000+2000
xmax = 69000+2000
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
time1=ds1.coords['time'].values/np.timedelta64(1, 's')
time=ds2.coords['time'].values/np.timedelta64(1, 's')
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
    dt=time[1]-time[0]
    print(dt)

    time_bin_labels = np.arange(12.4*60*60/2,time[-1]-20000,12.4*60*60)
    print(time_bin_labels)
    #time_bin = np.arange(0.,ds1.time[-1],freq='44660S')
    time_bin = pd.timedelta_range(0, periods=11,freq='44660S')
    print(time_bin)
    

    #depth mean velocity
    U0W=((ds1['UVEL']*ds1['drF']*ds1['hFacW']).sum('Z'))/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
    V0S=((ds1['VVEL']*ds1['drF']*ds1['hFacS']).sum('Z'))/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
    ds1['U0W']=xr.DataArray(U0W,coords=[time1,yc,xg],dims=['time','YC','XG'])
    ds1['V0S']=xr.DataArray(V0S,coords=[time1,yg,xc],dims=['time','YG','XC'])
    #baroclinic velocity
    ds1['upW']=(ds1['UVEL']-ds1['U0W'])*ds1['maskW']
    ds1['vpS']=(ds1['VVEL']-ds1['V0S'])*ds1['maskS']
    
    # horizontal dissipation
    hDisp = rhoNil*viscAh*(((grid.diff(ds1.UVEL* ds1.dyG , 'X', boundary='extrapolate')/ds1.rA)**2+(grid.diff(ds1.VVEL* ds1.dxG , 'Y', boundary='extrapolate')/ds1.rA)**2)*(ds1['drF']*ds1['hFacC'])).sum('Z')
    hDispbc1 = rhoNil*viscAh*(((grid.diff(ds1.upW* ds1.dyG , 'X', boundary='extrapolate')/ds1.rA)**2+(grid.diff(ds1.vpS* ds1.dxG , 'Y', boundary='extrapolate')/ds1.rA)**2)*(ds1['drF']*ds1['hFacC'])).sum('Z')
    hDispbc2 = rhoNil*viscAh*(((grid.diff(ds1.vpS* ds1.dyC , 'X', boundary='extrapolate')/ds1.rAz)**2+(grid.diff(ds1.upW* ds1.dxC , 'Y', boundary='extrapolate')/ds1.rAz)**2)*(ds1['drF']*ds1['hFacZ'])).sum('Z')
    hDispbt = rhoNil*viscAh*((grid.diff(ds1.U0W* ds1.dyG , 'X', boundary='extrapolate')/ds1.rA)**2+(grid.diff(ds1.V0S* ds1.dxG , 'Y', boundary='extrapolate')/ds1.rA)**2)
    print(hDispbc2)

    ta_hDisp = hDisp.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    ta_hDispbc1 = hDispbc1.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    ta_hDispbc2 = hDispbc2.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    ta_hDispbt = hDispbt.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    hDssp=(ta_hDisp*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    hDsspbc=(ta_hDispbc1*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6+(ta_hDispbc2*ds2['rAz']).sel(XG=xg[(xg > xmin) & (xg < xmax)],YG=yg[(yg > ymin) & (yg < ymax)]).sum(['XG','YG'])/1e6
    hDsspbt=(ta_hDispbt*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print('hDssp'+ str(hDssp.values))
    print('hDsspbc'+ str(hDsspbc.values))
    print('hDsspbt'+ str(hDsspbt.values))

    # dEdt
    Ebc=rhoNil*ds2['SDIAG5']
    dtEbc=np.gradient(Ebc.values,dt,axis=0)
    ddtEbc = xr.DataArray(dtEbc, coords=[ds2.time.values,yc,xc], dims=['time','YC', 'XC'])
    ta_dtEbc = ddtEbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    DtdE= (ta_dtEbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    
    # hdFbc
    uPbc=xr.DataArray(rhoNil*ds2['SDIAG6'], coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
    vPbc=xr.DataArray(rhoNil*ds2['SDIAG7'], coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
    uEbc=xr.DataArray(rhoNil*ds2['SDIAG8'], coords=[ds2.time.values,yc,xg], dims=['time','YC','XG'])
    vEbc=xr.DataArray(rhoNil*ds2['SDIAG9'], coords=[ds2.time.values,yg,xc], dims=['time','YG','XC'])
    #print(uPbc)
    #print(uEbc)
    Fxbc=uPbc+uEbc
    Fybc=vPbc+vEbc
    ta_Fxbc=Fxbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    ta_Fybc=Fybc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    #print(Fbc)
    hd_ta_Fbc=(grid.diff(ta_Fxbc*ds2['dyG'],'X',boundary='extrapolate')+grid.diff(ta_Fybc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']
    BCrad=(hd_ta_Fbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    fxdy=((ta_Fxbc*ds2['dyG']).isel(XG=ix[-1])-(ta_Fxbc*ds2['dyG']).isel(XG=ix[0])).sum('YC')/1e6
    # Flux only from pressure work
    if 0:
        ta_uPbc=uPbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
        ta_vPbc=vPbc.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
        hd_ta_uPbc=(grid.diff(ta_uPbc*ds2['dyG'],'X',boundary='extrapolate')+grid.diff(ta_vPbc*ds2['dxG'],'Y',boundary='extrapolate'))/ds2['rA']
        BCf_psw=(hd_ta_uPbc*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    
    # BC-BT conversion
    Conv=xr.DataArray(rhoNil*ds2['SDIAG10'], coords=[ds2.time.values,yc,xc], dims=['time','YC','XC'])
    print('Conv'+str(Conv))
    ta_Conv=Conv.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    print('ta_Conv'+str(ta_Conv))
    C=(ta_Conv*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print('C'+str(C))
    
    R=((ta_Conv-ta_dtEbc)*ds2['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6-fxdy
    print(R)

    # dissipation
    Dsp=-(rhoNil*ds1['KLeps']).integrate("Zl")
    print('Dsp'+str(Dsp))
    ta_dsp=Dsp.groupby_bins('time',time_bin,labels=time_bin_labels).mean()
    print('ta_dsp'+str(ta_dsp))
    dmax=2#np.amax(Dsp.values)
    dmin=10**(-10)
    D=(ta_dsp*ds1['rA']).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(['XC','YC'])/1e6
    print(D)
    
    RAT1=D/C*100
    RAT2=R/C*100
    print('rat1' +str(RAT1))

    ax2=host.twinx()
    par2=host.twinx()
    
    offset = 0
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",axes=par2,offset=(offset, 0))
    par2.axis["right"].toggle(all=True)

    host.set_ylim(-0.2,10)
    host.set_xlim(0,440000)
    

    host.set_ylabel("Energy Budget [MW]")
    host.set_xlabel("time [s]")
    par2.set_ylabel("Percentage [%]") 
 
    p1, = host.plot(DtdE.time_bins,DtdE.values,'o-',label="dE/dt")
    #BCrad.plot()
    host.plot(fxdy.time_bins,fxdy,'o-',label=r"$\nabla F_{bc}$")
    host.plot(C.time_bins,C,'o-',label="BT-BC Conversion")
    RL, = host.plot(R.time_bins,R,'o-',color='grey',label=r"Res=Conv-dE/dt-$\nabla F_{bc}$")
    DL, = host.plot(D.time_bins,D,'o-',label=r"Dissipation=$\int \rho \varepsilon dV$")
    host.plot(time1,np.zeros(ttlen),'k--')
    #host.plot(BCf_psw.time_bins,BCf_psw,'o-',label=r"$\nabla F (pressure)")
    host.plot(hDssp.time_bins,hDssp,'o-',label="horizontal Dssp")
    host.plot(hDsspbc.time_bins,hDsspbc,'o-',label="horizontal bc Dssp")
    host.plot(hDsspbt.time_bins,hDsspbt,'o-',label="horizontal bt Dssp")

    p2, = ax2.plot(RAT1.time_bins,RAT1,label="Disp/Conv",linestyle="--",color=DL.get_color())
    p3, = par2.plot(RAT2.time_bins,RAT2,label="Res/Conv",linestyle="--",color=RL.get_color())
    ax2.set_ylim(-5,100)
    par2.set_ylim(-5,100)

    host.legend()
    #plt.legend(('dE/dt',r'$\nabla F_{bc}$','BT-BC Conversion',r'Res=Conv-dE/dt-$\nabla F_{bc}$',r'Dissipation=$\int \rho \varepsilon dV$'))
    
    

    #plt.show()
    #plt.tight_layout()
    plt.savefig('./figs/EnergyBudget_hDssp_x27_t_tidal_p.png')
    #plt.show()


