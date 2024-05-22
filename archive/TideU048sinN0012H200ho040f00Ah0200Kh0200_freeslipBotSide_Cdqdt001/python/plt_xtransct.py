from xmitgcm import open_mdsdataset
import xgcm
import psyplot.project as psy
import os
import matplotlib.pyplot as plt
from matplotlib import cm
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
Grid = xgcm.Grid(ds1, periodic=False)

t = 0
h0 = 140.
H = 200.
f0 = 1.e-4
g = 9.8
u0=0.12

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


xmin = 30+2
xmax = 54+2
vmax = 1.0
vmin = -1.0
wmax = 0.2
wmin = -0.2
tmax = math.floor(np.amax(tRef))
tmin = math.floor(np.amin(tRef))
numcolt=15
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print(tmin)


cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=numcolv)
Tlevels = np.linspace(tmin, tmax, num=numcolt)
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

   
xg=ds1.coords["XG"]
xc=ds1.coords["XC"]
yg=ds1.coords["YG"]
yc=ds1.coords["YC"]
z=ds1.coords["Z"]
time=ds1.coords["time"].values/np.timedelta64(1, 's')/3600
print(ds1)
print(xc[185:246])
print(xc[-2:])
print(ds1.XC.mean())

ds1=ds1.assign_coords(XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.))
ds1=ds1.assign_coords(XG0 = (ds1.XG-ds1.XC.mean())/1000.)
ds1=ds1.assign_coords(YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.))
ds1=ds1.assign_coords(YG0 = (ds1.YG-ds1.YC.mean())/1000.)
yc0 = (ds1.YC/1000.-ds1.YC.mean()/1000.)
xc0 = (ds1.XC/1000.-ds1.XC.mean()/1000.)

fig = plt.figure(figsize=(5.5,12))
grid = plt.GridSpec(6, 2, hspace=0.2,wspace=0.22, height_ratios=[1,1,1, 1,1,0.3], width_ratios=[1,1])

# select plotting slices
lcx1=44010
lcx2=44850

(lx1,ix1) = min((abs(x-lcx1), ix) for ix, x in enumerate(xc.values) )
print (lx1,ix1)

(lx2,ix2) = min((abs(x-lcx2), ix) for ix, x in enumerate(xc.values) )
print (lx2,ix2)

ix=[ix1,ix2]
print(ix)

it=range(145,155,2)

from itertools import product
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
for (xx,tt) in product(ix,it):
    print(xx,tt)
    ixx=ix.index(xx)
    itt=it.index(tt)
    ax=fig.add_subplot(grid[itt,ixx])
    U=ds1['UVEL'].isel(time=tt)
    T=ds1['THETA'].isel(time=tt).where(ds1['maskC']!=0,np.nan)
    Dep=-ds1['Depth']
    UC=Grid.interp(U,'X',boundary='extrapolate')
    pcm=ax.pcolormesh(ds1['YC0'],ds1['Z'],UC.where(ds1['maskC'] !=0,np.nan).isel(XC=xx),vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
    ax.contour(ds1['YC0'], ds1['Z'],T.isel(XC=xx),np.sort(tRef[::4]),colors='0.3',linewidths=1)
    ax.plot(ds1['YC0'],Dep.isel(XC=xx),lw=1.2,color='k')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-152,1])

    if itt==0:
        ax.set_title('XC= %1.1f km' %xc0[xx])
    if itt!=len(it)-1:
        ax.tick_params(labelbottom=False)


    if ixx==0:
        bot_inset_ax = inset_axes(ax,width="33%", height="40%",loc=8)        
        bot_inset_ax.plot(time,np.zeros(ttlen),color='gray',linestyle='--')
        bot_inset_ax.plot(time,ds1['UVEL'].isel(YC=2, XG=0, Z=0),label='U(t,0,0,0)')
        bot_inset_ax.set_xlim(73, 83) #edit
        bot_inset_ax.set_ylim(-u0-0.03, u0+0.03)
        print(ds1['UVEL'].isel(time=tt,YC=2, XG=0, Z=0).values)
        point, = bot_inset_ax.plot([ds1['time'].isel(time=tt).values/np.timedelta64(1, 's')/3600],[ds1['UVEL'].isel(time=tt,YC=2, XG=0, Z=0)], 'go', color='firebrick',ms=5)
        bot_inset_ax.tick_params(labelleft=False,labelbottom=False)
        if itt==len(it)-1:
            ax.set_ylabel('Z [m]')
            ax.set_xlabel('Y [km]')
    else:
        ax.tick_params(labelleft=False)

#common colorbar
cax2 = fig.add_subplot(grid[-1,:])
cax2.set_visible(False)
cbar=fig.colorbar(pcm, ax=cax2,extend='both',orientation="horizontal")
cbar.ax.set_xlabel('U [$ms^{-1}$]')

    
#plt.show()
plt.savefig('./figsMag2/xtransct0_900m_t145_155.png' )

#plt.show()


