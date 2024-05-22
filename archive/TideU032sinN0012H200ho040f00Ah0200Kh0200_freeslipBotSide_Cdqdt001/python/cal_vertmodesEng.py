from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
import xarray as xr
import string
import dask.array as da
import vertmodes

import warnings

warnings.filterwarnings('ignore')

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
refTemp=tRef[0]

rho2=rhoNil*(1-(alpha*(tRef-refTemp)))
rhoS=np.roll(rho2,1)
N2=g/rhoNil*(rho2-rhoS)/ds1['drF'].values
N2[0]=g/rhoNil*(rhoS[0]-rho2[0])/ds1['drF'][0]
print(N2)

xmin = 34000
xmax = 50000
ymin = 0000
ymax = 3000
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
z=ds1.coords['Z']
zl=ds1.coords['Zl']
dz=-np.median(np.diff(z))

psi,phi,ce,zph = vertmodes.vertModes(N2,dz)

Nm=37
if 0:
    fig,axs=plt.subplots(1,2)
    for nn in range(Nm):
        axs[0].plot(psi[:,nn],-zph)
        axs[1].plot(phi[:,nn],-zph)
    axs[0].set_ylabel('DEPTH [m]')
    axs[0].set_xlabel('$\phi$: horizontal structure')
    axs[0].legend(('mode 0','mode 1','mode 2','mode 3'))
    axs[1].set_xlabel('$\psi$: vertical structure')
    fig.savefig('./figs/normodestructure.png')
    plt.close(fig)

ds1['PW'] = xr.DataArray(0.5*(ds1['PHIHYD'].roll(XC=1)+ds1['PHIHYD']).data,coords=[time1,z,yc,xg],dims=('time','Z','YC','XG'))
ds1['PS'] = xr.DataArray(0.5*(ds1['PHIHYD'].roll(YC=1)+ds1['PHIHYD']).data,coords=[time1,z,yg,xc],dims=('time','Z','YG','XC'))
print(ds1.PW)
print(ds1.PS)

ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])
for i in range(Nm):
    exec('psi%s=psi[:,i]*(ds1.drF.sum("Z").values)**0.5'%i)
    exec('phi%s=phi[:,i]*(ds1.drF.sum("Z").values)**0.5'%i)
    exec('ds1["psi%s"]=xr.DataArray(psi%s,coords=[z],dims=["Z"])'%(i,i))
    exec('ds1["phi%s"]=xr.DataArray(phi%s,coords=[z],dims=["Z"])'%(i,i))

time_bin_labels = np.arange(12.4*60*60/2,time.values[-1]-20000,12.4*60*60)
print(time_bin_labels)
for i in range(Nm):
    exec('BCrad%s_td=np.zeros(10)'%i)

for td in range(0,10):
    print(td)
    dt=time.values[1]-time.values[0]
    print(dt)
    
    T=td

    t_st=int(12.4*60*60*T/dt+1)
    t_en=int(12.4*60*60*(T+1)/dt+1)
    timestd=np.arange(12.4*60*60*T+dt,12.4*60*60*(T+1)+dt,dt)
    
    for i in range(Nm):
        exec('Fx%s_t=np.zeros((t_en-t_st,len(yc),len(xg)),dtype=float)'%i)
        exec('Fy%s_t=np.zeros((t_en-t_st,len(yg),len(xc)),dtype=float)'%i)

    for tt in range(t_st,t_en):
        print(tt)
        U0=(ds1['UVEL'].isel(time=tt)*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
        V0=(ds1['VVEL'].isel(time=tt)*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
        PW0=(ds1.PW.isel(time=tt)*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
        PS0=(ds1.PS.isel(time=tt)*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
        
        up=(ds1.UVEL.isel(time=tt)-U0).where(ds1.maskW!=0,np.nan)
        vp=(ds1.VVEL.isel(time=tt)-V0).where(ds1.maskS!=0,np.nan)
        ppw=(ds1.PW.isel(time=tt)-PW0).where(ds1.maskW!=0,np.nan)
        pps=(ds1.PS.isel(time=tt)-PS0).where(ds1.maskS!=0,np.nan)
        del U0,V0,PW0,PS0


        for i in range(Nm):
            exec('u%s=(up*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'% (i,i))
            exec('pw%s=(ppw*ds1.psi%s*ds1.drF).sum("Z") / ds1.drF.sum("Z")'%(i,i))
            exec('v%s=(vp*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'%(i,i))
            exec('ps%s=(pps*ds1.psi%s*ds1.drF).sum("Z") / ds1.drF.sum("Z")'%(i,i))
            exec('fx%s=(rhoNil*u%s*pw%s*ds1.psi%s*ds1.psi%s*ds1.drF*ds1.hFacW*ds1.maskW).sum("Z")'%(i,i,i,i,i))
            exec('fy%s=(rhoNil*v%s*ps%s*ds1.psi%s*ds1.psi%s*ds1.drF*ds1.hFacS*ds1.maskS).sum("Z")'%(i,i,i,i,i))
            exec('Fx%s_t[tt-t_st,:,:]=fx%s'%(i,i))
            exec('Fy%s_t[tt-t_st,:,:]=fy%s'%(i,i))
            exec('del u%s,pw%s,v%s,ps%s'%(i,i,i,i))
    BCrad=0
    del up,vp,ppw,pps

    for i in range(Nm):
        exec('Fx%s=xr.DataArray(Fx%s_t,coords=[timestd,yc,xg],dims=("time","YC","XG"))'%(i,i))
        exec('Fy%s=xr.DataArray(Fy%s_t,coords=[timestd,yg,xc],dims=("time","YG","XC"))'%(i,i))
        exec('ta_Fx%s=xr.DataArray(Fx%s_t.mean(axis=0),coords=[yc,xg],dims=["YC","XG"])'%(i,i))
        exec('ta_Fy%s=xr.DataArray(Fy%s_t.mean(axis=0),coords=[yg,xc],dims=["YG","XC"])'%(i,i))
        
        exec('hd_ta_F%s= (grid.diff(ta_Fx%s*ds1["dyG"],"X",boundary="extrapolate")+grid.diff(ta_Fy%s*ds2["dxG"],"Y",boundary="extrapolate"))/ds1["rA"]'%(i,i,i))
            
        exec('BCrad%s   = (hd_ta_F%s*ds1["rA"]).sel(XC=xc[(xc > xmin) & (xc < xmax)],YC=yc[(yc > ymin) & (yc < ymax)]).sum(["XC","YC"])/1e6'%(i,i))
        exec('del Fx%s,Fy%s,ta_Fx%s,ta_Fy%s,hd_ta_F%s'%(i,i,i,i,i))
        exec('print("BCrad%s:" +str(BCrad%s.values))'%(i,i))
        exec('BCrad=BCrad+BCrad%s.values'%i)
        exec('BCrad%s_td[td]=BCrad%s.values'%(i,i))
    
    print('sum BCrad_n:' + str(BCrad))


print(BCrad0_td)
print(BCrad1_td)



if 1:
    f, ax =plt.subplots(figsize=(15,6))
    for i in range(10):
        exec('ax.plot(time_bin_labels,BCrad%s_td,label="mode%s")'%(i,i+1))

    #ax.set_ylim(-0.2,10)
    ax.set_xlim(0,400000)
    
    ax.set_ylabel("Energy Budget [MW]")
    ax.set_xlabel("time [s]")
    ax.legend()
    #plt.tight_layout()
    plt.savefig("./figs/EnergyBudget_BCRad_modes.png")
    plt.clf()
    plt.close(f)


