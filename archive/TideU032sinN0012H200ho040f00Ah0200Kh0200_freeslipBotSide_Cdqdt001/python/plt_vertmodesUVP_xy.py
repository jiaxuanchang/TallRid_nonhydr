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

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])

ds1.to_netcdf('ds1.nc')
ds2.to_netcdf('ds2.nc')
stop
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

#ds1.coords['time']=ds1.coords["time"]/3600
#ds1.coords['XG'] = ds1.coords['XG']/1000
#ds1.coords['XC'] = ds1.coords['XC']/1000
time1=ds1.coords['time']/3600
time=ds2.coords['time']
xc=ds1.coords['XC']
xg=ds2.coords['XG']
yc=ds2.coords['YC']
yg=ds2.coords['YG']
z=ds1.coords['Z']
zl=ds1.coords['Zl']
dz=-np.median(np.diff(z))

psi,phi,ce,zph = vertmodes.vertModes(N2,dz)

ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])
for i in range(4):
    exec('psi%s=psi[:,i]*(ds1.drF.sum("Z").values)**0.5'%i)
    exec('phi%s=phi[:,i]*(ds1.drF.sum("Z").values)**0.5'%i)
    exec('ds1["psi%s"]=xr.DataArray(psi%s,coords=[z],dims=["Z"])'%(i,i))
    exec('ds1["phi%s"]=xr.DataArray(phi%s,coords=[z],dims=["Z"])'%(i,i))
    exec('u%s=(ds1.UVEL*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'% (i,i))

print(ds1.psi0)
print(u0)
u_nm=u0.isel(YC=28,time=100,XG=194).values*ds1.psi0+u1.isel(YC=28,time=100,XG=194).values*ds1.psi1+u2.isel(YC=28,time=100,XG=194).values*ds1.psi2+u3.isel(YC=28,time=100,XG=194).values*psi3
print(u_nm.values)
print(ds1.UVEL.isel(YC=28,time=100,XG=194).values)
#plt.plot(ds1.UVEL.isel(YC=28,time=100,XG=194))
#plt.plot(u_nm)
#plt.show()

print(ds1['phi1'])

time_bin_labels = np.arange(12.4*60*60/2,time.values[-1]-20000,12.4*60*60)
print(time_bin_labels)

for i in range(4):
    # flat bottom or not?
    exec('u%s=(ds1.UVEL*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'% (i,i))
    exec('p%s=(ds1.PHIHYD*ds1.psi%s*ds1.drF).sum("Z") / ds1.drF.sum("Z")'%(i,i))


for t in range(100,150,5):

    f, ax = plt.subplots(2,6, figsize=(20,9), sharex=True, sharey=True)
    ds1.UVEL.isel(time=t).mean('Z').plot(ax=ax[0,0],vmin=-0.8,vmax=0.8,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
    ds1.Depth.plot.contour(ax=ax[0,0],lw=0.7,levels=[0,200],colors='grey')
    ax[0,0].set_title('depth-averaged U')
    ax[0,0].set_xlim(20000,64000)
    
    (ds1.UVEL.isel(time=t,Z=0)-ds1.UVEL.isel(time=t).mean('Z')).plot(ax=ax[0,1],vmin=-0.8,vmax=0.8,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
    ds1.Depth.plot.contour(ax=ax[0,1],levels=[0,200],colors='grey')
    ax[0,1].set_title('Usuf - depth-averaged U')
    ax[0,1].set_xlim(20000,64000)
    
    ds1.PHIHYD.isel(time=t).mean('Z').plot(ax=ax[1,0],vmin=-6,vmax=6,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
    ds1.Depth.plot.contour(ax=ax[1,0],levels=[0,200],colors='grey')
    ax[1,0].set_title('depth-averaged P')
    ax[1,0].set_xlim(20000,64000)
    
    (ds1.PHIHYD.isel(time=t,Z=0)-ds1.PHIHYD.isel(time=t).mean('Z')).plot(ax=ax[1,1],vmin=-6,vmax=6,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
    ds1.Depth.plot.contour(ax=ax[1,1],levels=[0,200],colors='grey')
    ax[1,1].set_title('Psuf - depth-averaged P')
    ax[1,1].set_xlim(20000,64000)


    for i in range(4):
        exec('print(u%s)'%i)
        exec('u%s.isel(time=t).plot(ax=ax[0,i+2],vmin=-0.8,vmax=0.8,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})'%i)
        ds1.Depth.plot.contour(ax=ax[0,i+2],levels=[0,200],colors='grey')
        exec('ax[0,i+2].set_title("u%s")'%i)
        ax[0,i+2].set_xlim(20000,64000)
        exec('p%s.isel(time=t).plot(ax=ax[1,i+2],vmin=-6,vmax=6,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})'%i)
        ds1.Depth.plot.contour(ax=ax[1,i+2],levels=[0,200],colors='grey')
        exec('ax[1,i+2].set_title("p%s")'%i)
        ax[1,i+2].set_xlim(20000,64000)
    
    f.suptitle("t= %2.2f hrs" %time1[t], size=16)
    plt.tight_layout()
    plt.savefig('figs/modesUP_t%d.png'%t)
    plt.clf()
    plt.close()


