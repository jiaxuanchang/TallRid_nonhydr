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

fig,axes = plt.subplots(2, 6, figsize=(11,6),sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.2,top=0.95,hspace=0.22,wspace=0.2)
axes=axes.flatten()

# select plotting slices
sy=[1525] 
iy=[]
it=range(139,163,2)

for i in range(len(sy)):
    if sy[i] in yc:
        iy.append(list(yc.data).index(sy[i]))
        print(iy)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
for ii,tt in enumerate(it):
    U=ds1['UVEL'].isel(time=tt)
    T=ds1['THETA'].isel(time=tt).where(ds1['maskC']!=0,np.nan)
    Dep=-ds1['Depth']
    UC=Grid.interp(U,'X',boundary='extrapolate')
    axes[ii].plot([0,0],[1,1])
    uplt=np.squeeze(U.where(ds1['maskW'] !=0,np.nan).isel(YC=iy))
    pcm=axes[ii].pcolormesh(ds1['XG0'],ds1['Z'],uplt,vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
    tplt=np.squeeze(T.isel(YC=iy))
    axes[ii].contour(ds1['XC0'], ds1['Z'],tplt,np.sort(tRef[::4]),colors='0.3',linewidths=1)
    axes[ii].plot(ds1['XC0'],np.squeeze(Dep.isel(YC=iy)),lw=1.2,color='k')
    axes[ii].set_xlim([-1,1])
    axes[ii].set_ylim([-121,1])
    print(ii)
    if ii==0:
        axes[ii].set_ylabel('Z [m]')
        print(1)
    if ii==6:
        axes[ii].set_ylabel('Z [m]')
        axes[ii].set_xlabel('X [km]')

    bot_inset_ax = inset_axes(axes[ii],width="36%", height="30%",loc=8)        
    bot_inset_ax.plot(time,np.zeros(ttlen),color='gray',linestyle='--')
    bot_inset_ax.plot(time,ds1['UVEL'].isel(YC=2, XG=0, Z=0),label='U(t,0,0,0)')
    bot_inset_ax.set_xlim(70, 83) #edit
    bot_inset_ax.set_ylim(-u0-0.03, u0+0.03)
    print(ds1['UVEL'].isel(time=tt,YC=2, XG=0, Z=0).values)
    point, = bot_inset_ax.plot([ds1['time'].isel(time=tt).values/np.timedelta64(1, 's')/3600],[ds1['UVEL'].isel(time=tt,YC=2, XG=0, Z=0)], 'go', color='firebrick',ms=5)
    bot_inset_ax.tick_params(labelleft=False,labelbottom=False)

#common colorbar
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#ax_divider = make_axes_locatable(axes[7:11])
#cax=ax_divider.append_axes("bottom",size="7%", pad="2%")
cax = fig.add_subplot(position=[axes[7].get_position().x0
    ,axes[7].get_position().y0-0.1
    ,axes[10].get_position().x1-axes[7].get_position().x0,0.03])
cbar=fig.colorbar(pcm,cax=cax,extend='both',orientation="horizontal")
cbar.ax.set_xlabel('U [$ms^{-1}$]')

    
#plt.show()
plt.savefig('./figsMag2/Uytransct0_t139_161.png' )

#plt.show()


