from xmitgcm import open_mdsdataset
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import math

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds = open_mdsdataset(data_dir, geometry='cartesian', endian='<')

t = 0
#h0 = 340.
#H = float(currentDirectory[-15:-12])
#N0 = float(currentDirectory[-20:-16])/1e6
#Fr=int(currentDirectory[-26:-22])/1e4
#U0=Fr*N0*h0
f0 = 1.e-4
g = 9.8

u0=0.2
om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
#dz=H/nz 
Tref=23#35+np.cumsum(N0**2/g/alpha*(-H))

xmin = 38
xmax = 46
vmax = 1
vmin = -1
wmax = 0.2
wmin = -0.2
tmax = 35
tmin = math.floor(Tref)
numcolt=21
numcolv=21
                                        
ttlen=len(ds.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(Tref))
print(tmin)


cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=numcolv)
Tlevels = np.linspace(tmin, tmax, num=numcolt)
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

   
xg=ds.coords["XG"]
xc=ds.coords["XC"]
time=ds.coords["time"]/3600
fig = plt.figure(figsize=(12, 8))
grid = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1,0.02])

for t in range(ttlen):
    fig.clf()
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[0,1])
    #print(ds['UVEL'].isel(time=0,YC=0))
    U=ds['UVEL'].isel(time=t, YC=0)
    U=U.sel(XG=xg[(xg > xmin*1e3) & (xg < xmax*1e3)])
    T=ds['THETA'].isel(time=t, YC=0)
    #print(ds['Depth'].isel(YC=0))
    Dep=-ds['Depth'].sel(XC=xc[(xc > xmin*1e3) & (xc < xmax*1e3)])
    U.plot(ax=ax1,levels=levels,cmap='RdBu_r',cbar_ax=ax3)
    T.plot.contour(ax=ax1,levels=Tlevels,colors='0.3')
    Dep.plot(ax=ax1,color='k')
    ax1.set_ylim(-200,0)
    ax1.set_title('Time= %2.2f hrs' % time[t] )
    
    ax2.plot(time,np.zeros(ttlen),color='gray',linestyle='--')   
    ax2.plot(time,ds['UVEL'].isel(YC=0, XG=0, Z=0),label='U(t,0,0,0)')
    ax2.plot(time,u0*np.cos(om*ds['time']),label='Uinput')
    ax2.set_xlim(min(time), max(time))
    ax2.set_ylim(-u0-0.1, u0+0.1)
    point, = ax2.plot([ds['time'].isel(time=t)/3600],[ds['UVEL'].isel(time=t,YC=0, XG=0, Z=0)], 'go', color='red',ms=10)
    ax2.set_xlabel('time [hr]')
    ax2.set_ylabel('U [m/s]')
    ax2.set_title('Uinput vs U[t,0,0,0]', loc='left')
    fig.savefig('./figsMag/UT_tide_xz_i%03d.png' %t)

#plt.show()
