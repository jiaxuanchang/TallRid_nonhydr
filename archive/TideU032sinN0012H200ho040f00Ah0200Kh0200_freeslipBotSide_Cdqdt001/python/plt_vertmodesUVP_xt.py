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
time=ds1.coords['time']
xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
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
    exec('del psi%s, phi%s'%(i,i))   

print(ds1.psi0)
#print(u0)
#u_nm=u0.isel(YC=28,time=100,XG=194).values*ds1.psi0+u1.isel(YC=28,time=100,XG=194).values*ds1.psi1+u2.isel(YC=28,time=100,XG=194).values*ds1.psi2+u3.isel(YC=28,time=100,XG=194).values*psi3
#print(u_nm.values)
#print(ds1.UVEL.isel(YC=28,time=100,XG=194).values)
#plt.plot(ds1.UVEL.isel(YC=28,time=100,XG=194))
#plt.plot(u_nm)
#plt.show()

print(ds1['phi1'])

time_bin_labels = np.arange(12.4*60*60/2,time.values[-1]-20000,12.4*60*60)
print(time_bin_labels)


if 0:
    for iy in range(30,40,5):
        f, ax = plt.subplots(1,5, figsize=(20,9), sharex=True, sharey=True)
        
        (ds1.UVEL.isel(YC=iy,Z=0)-ds1.UVEL.isel(YC=iy).mean('Z')).plot(ax=ax[0],vmin=-0.8,vmax=0.8,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
        ax[0].plot([42010,80000],[0,(80000-42010)/ce[0]],'--')
        ax[0].plot([42010,80000],[0,(80000-42010)/ce[1]],'--')
        ax[0].plot([42010,80000],[0,(80000-42010)/ce[2]],'--')
        ax[0].plot([42010,80000],[0,(80000-42010)/ce[3]],'--')
        ax[0].set_title('Usuf - depth-averaged U')
        ax[0].set_xlim(20000,64000)
        ax[0].set_ylim(0,300000)
        up=(ds1.UVEL.isel(YC=iy)-ds1.UVEL.isel(YC=iy).mean(dim="Z"))

        for i in range(4):
            exec('u%s=(up*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'% (i,i))
            print(i)
            exec('u%s.plot(ax=ax[i+1],vmin=-0.8,vmax=0.8,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})'%i)
            exec('ax[i+1].set_title("u%s")'%(i+1))
            exec('ax[i+1].plot([42010,80000],[0,(80000-42010)/ce[i]],"--")')
            ax[i+1].set_xlim(20000,64000)
            ax[i+1].set_ylim(0,300000)
            exec('del u%s'%i)

        f.suptitle("YC= %d m" %yc[iy], size=16)
        plt.tight_layout()
        plt.savefig('figs/modesU_xt_iy%d.png'%iy)
        print(iy)
        plt.clf()
        plt.close(f)

if 1:
    for iy in range(30,40,5):
        
        f, ax = plt.subplots(1,5, figsize=(20,9), sharex=True, sharey=True)
        
        pp=(ds1.PHIHYD.isel(YC=iy)-ds1.PHIHYD.isel(YC=iy).mean(dim="Z"))
        (rhoNil*ds1.PHIHYD.isel(YC=iy,Z=0)).plot(ax=ax[0],vmin=-800,vmax=800,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})
        ax[0].set_title('Psuf')
        ax[0].set_xlim(20000,64000)
        ax[0].set_ylim(0,300000)


        for i in range(4):
            print(i)
            exec('p%s=(pp*ds1.psi%s*ds1.drF).sum("Z")/ ds1.drF.sum("Z")'% (i,i))
            exec('p%s.plot(ax=ax[i+1],vmin=-0.08,vmax=0.08,cmap="RdBu_r",cbar_kwargs={"aspect": 40,"label": ""})'%i)
            exec('ax[i+1].set_title("p%s")'%(i+1))
            ax[i+1].set_xlim(20000,64000)
            ax[i+1].set_ylim(0,300000)
            exec('del p%s'%i)
        
        f.suptitle("YC= %d m" %yc[iy], size=16)
        plt.tight_layout()
        plt.savefig('figs/modesP_xt_iy%d.png'%iy)
        plt.clf()
        plt.close(f)
        print(iy)

