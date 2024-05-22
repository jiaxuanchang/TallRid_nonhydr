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

ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])

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


xmin = 38
xmax = 46
vmax = 0.6
vmin = -0.6
wmax = 0.2
wmin = -0.2
tmax = math.floor(np.amax(tRef))
tmin = math.floor(np.amin(tRef))
numcolt=21
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
time=ds1.coords["time"]/3600
fig=plt.figure(figsize=(15,8))

for t in range(100,110,5):#range(ttlen):
    fig.clf()
    #print(ds['UVEL'].isel(time=0,YC=0))
    U=ds1['UVEL'].isel(time=t,YC=slice(0,75,5))
    U=U.sel(XG=xg[(xg > xmin*1e3) & (xg < xmax*1e3)])
    T=ds1['THETA'].isel(time=t)
    #print(ds['Depth'].isel(YC=0))
    Dep=-ds1['Depth'].sel(XC=xc[(xc > xmin*1e3) & (xc < xmax*1e3)])
    fig.suptitle("t= %2.2f hrs" %time[t])
    s=range(0,75,5)
    g=U.plot(levels=levels,y='Z',x='XG',col='YC',col_wrap=5,cmap='RdBu_r')
    for iy, ax in enumerate(g.axes.flat):
        T.isel(YC=s[iy]).plot.contour(ax=ax,x='XC',y='Z',levels=Tlevels,colors='0.3')
        Dep.isel(YC=s[iy]).plot(ax=ax,color='k')
        ax.set_ylim(-200,0)
        #ax.set_title('Y= %05d km' %yc[iy] )
    #plt.show()
    fig.suptitle("t= %2.2f hrs" %time[t], size=16)
    plt.savefig('./figsMag/UT_xz_i%03d.png' %t)

#plt.show()


