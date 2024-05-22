from xmitgcm import open_mdsdataset
import xgcm
import psyplot.project as psy
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import math
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
h0 = 140.
H = 200.
f0 = 1.e-4
g = 9.8

om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
dz=H/nz 

tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.
refTemp=tRef[0]
print('refTemp='+ str(refTemp))
print('tRef='+ str(tRef))

xmin = -3
xmax = 3
vmax = 5e-3
vmin = -5e-3
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )

cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=numcolv)
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

print(ds1)

xg=ds1.coords["XG"]
xc=ds1.coords["XC"]
yg=ds1.coords["YG"]
yc=ds1.coords["YC"]
time=ds1.coords["time"]/3600
XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.).values
YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.).values
XG0 = (ds1.XG/1000.-ds1.XC.mean()/1000.).values
YG0 = (ds1.YG/1000.-ds1.YC.mean()/1000.).values
Z=ds1.coords["Z"]

windowsize=5


# plotting
widths = [1, 1, 1, 1, 0.05]
gs_kw = dict(width_ratios=widths)
fig, axes = plt.subplots(figsize=(16,6),nrows=2, ncols=5, gridspec_kw=gs_kw)
fig.subplots_adjust(top=0.9,left=0.06,right=0.92)

# plot vorticity at z = -42.5 m (index=8)
for t in range(106,110,1):
    U=ds1['UVEL'].isel(time=t,Z=8)
    V=ds1['VVEL'].isel(time=t,Z=8)
    maskZ = grid.interp(ds1.hFacS, 'X', boundary='extend')

    zeta = (-grid.diff(U * ds1['dxC'], 'Y', boundary='extend') + grid.diff(V * ds1['dyC'], 'X',boundary='extend'))/ds1['rAz']
    zeta = xr.DataArray(zeta,coords=[YG0,XG0],dims=["YG0","XG0"])
    maskZ = xr.DataArray(maskZ.isel(Z=8),coords=[YG0,XG0],dims=["YG0","XG0"])
    zeta['XG0'].attrs["units"] = "km"
    zeta['YG0'].attrs["units"] = "km"
    maskZ['XG0'].attrs["units"] = "km"
    maskZ['YG0'].attrs["units"] = "km"

    print(zeta)
    print(maskZ)

    PHIBOT = xr.DataArray(ds1['PHIBOT'].isel(time=t),coords=[YC0,XC0],dims=["YC0","XC0"])
    PHIBOT['XC0'].attrs["units"] = "km"
    PHIBOT['YC0'].attrs["units"] = "km"

    zeta = zeta.sel(XG0=XG0[(XG0 > xmin) & (XG0 < xmax)])
    maskZ = maskZ.sel(XG0=XG0[(XG0 > xmin) & (XG0 < xmax)])

    zetachunked=zeta.chunk(chunks=(60,400))

    sm_zeta = zetachunked.rolling(XG0=windowsize,center=True).mean()
    sm_zeta = sm_zeta.rolling(YG0=windowsize,center=True).mean()
    
    if t==109:
        pm1=sm_zeta.where(maskZ).plot(ax=axes[0,t-106],levels=levels,cmap='RdYlBu_r',cbar_ax=axes[0,4])
    else:
        pm1=sm_zeta.where(maskZ).plot(ax=axes[0,t-106],levels=levels,cmap='RdYlBu_r',add_colorbar=False)
    if t==106:
        axes[0,t-106].text(-2.8,1,r'$ \zeta [s^{-1}]$',size=14)
    else:
        axes[0,t-106].set_xticklabels([])
        axes[0,t-106].set_xlabel('')
        axes[0,t-106].set_yticklabels([])
        axes[0,t-106].set_ylabel('')

    axes[0,t-106].set_xlim(xmin,xmax)
    axes[0,t-106].set_title("t= %2.2f hrs, Z= -42.5 m" %time[t], size=16)
    
    if t==109:
        pm2=PHIBOT.plot(ax=axes[1,t-106],vmax=0.6,vmin=-0.6,cmap='RdYlBu_r',cbar_ax=axes[1,4])
    else:
        pm2=PHIBOT.plot(ax=axes[1,t-106],vmax=0.6,vmin=-0.6,cmap='RdYlBu_r',add_colorbar=False)

    axes[1,t-106].set_xlim(xmin,xmax)
    if t==106:
        axes[1,t-106].text(-2.8,1,r'$\frac{P^{\prime}_{bot}}{\rho_0} [m^2s^{-2}]$',size=14)
    else:
        axes[1,t-106].set_xticklabels([])
        axes[1,t-106].set_xlabel('')
        axes[1,t-106].set_yticklabels([])
        axes[1,t-106].set_ylabel('')


plt.savefig('./figsMag2/vortices.png')

